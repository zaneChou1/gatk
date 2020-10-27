package org.broadinstitute.hellbender.tools.sv;

import com.google.common.collect.Lists;
import htsjdk.samtools.util.Locatable;
import htsjdk.variant.variantcontext.*;
import htsjdk.variant.variantcontext.writer.VariantContextWriter;
import htsjdk.variant.vcf.*;
import org.apache.commons.math3.util.FastMath;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.barclay.argparser.Hidden;
import org.broadinstitute.barclay.help.DocumentedFeature;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.cmdline.programgroups.StructuralVariantDiscoveryProgramGroup;
import org.broadinstitute.hellbender.engine.*;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.genotyper.AlleleList;
import org.broadinstitute.hellbender.utils.genotyper.IndexedAlleleList;
import org.broadinstitute.hellbender.utils.io.IOUtils;
import org.broadinstitute.hellbender.utils.python.StreamingPythonScriptExecutor;
import org.broadinstitute.hellbender.utils.runtime.AsynchronousStreamWriter;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Annotate a VCF with scores from a Convolutional Neural Network (CNN).
 *
 * This tool streams variants and their reference context to a python program,
 * which evaluates a pre-trained neural network on each variant.
 * The default models were trained on single-sample VCFs.
 * The default model should not be used on VCFs with annotations from joint call-sets.
 *
 * <h3>1D Model with pre-trained architecture</h3>
 *
 * <pre>
 * gatk CNNScoreVariants \
 *   -V vcf_to_annotate.vcf.gz \
 *   -R reference.fasta \
 *   -O annotated.vcf
 * </pre>
 *
 */
@DocumentedFeature
@CommandLineProgramProperties(
        summary = SVTrainGenotyping.USAGE_SUMMARY,
        oneLineSummary = SVTrainGenotyping.USAGE_ONE_LINE_SUMMARY,
        programGroup = StructuralVariantDiscoveryProgramGroup.class
)

public class SVGenotype extends TwoPassVariantWalker {

    private final static String NL = String.format("%n");
    private static final String COLUMN_SEPARATOR = "\t";
    private static final String FIRST_DIM_SEPARATOR = ";";
    private static final String SECOND_DIM_SEPARATOR = ",";

    public static final String COPY_NUMBER_FIELD = "CN";
    public static final String PAIRED_END_PROB_FIELD = "PPE";
    public static final String FIRST_SPLIT_READ_PROB_FIELD = "PSR1";
    public static final String SECOND_SPLIT_READ_PROB_FIELD = "PSR2";
    public static final String DEPTH_SUPPORT_PROB_FIELD = "PRD";
    public static final String PAIRED_END_BACKGROUND_FIELD = "EPE";
    public static final String FIRST_SPLIT_READ_BACKGROUND_FIELD = "ESR1";
    public static final String SECOND_SPLIT_READ_BACKGROUND_FIELD = "ESR2";
    public static final String PAIRED_END_MEAN_BIAS_FIELD = "PHI_PE";
    public static final String FIRST_SPLIT_READ_MEAN_BIAS_FIELD = "PHI_SR1";
    public static final String SECOND_SPLIT_READ_MEAN_BIAS_FIELD = "PHI_SR2";
    public static final String HARDY_WEINBERG_Q_FIELD = "ETA_Q";
    public static final String HARDY_WEINBERG_R_FIELD = "ETA_R";

    // For missing data
    private static final int MAX_GENOTYPE_PL = 999;
    private static final int DEFAULT_GENOTYPE_PL = MAX_GENOTYPE_PL;
    private static final double MIN_GENOTYPE_PROB = Math.max(Math.pow(10., -MAX_GENOTYPE_PL), Double.MIN_VALUE);
    private static final double MIN_SITE_LOG10_PROB_NO_VAR = -99.9;

    static final String USAGE_ONE_LINE_SUMMARY = "Run model to genotype structural variants";
    static final String USAGE_SUMMARY = "Run model to genotype structural variants";

    @Argument(fullName = StandardArgumentDefinitions.OUTPUT_LONG_NAME,
            shortName = StandardArgumentDefinitions.OUTPUT_SHORT_NAME,
            doc = "Output VCF")
    private GATKPath outputVcf;

    @Argument(fullName = "intermediates-dir", doc = "Intermediate file directory")
    private GATKPath intermediatesDir;

    @Argument(fullName = "model-name", doc = "Model name")
    private String modelName;

    @Argument(fullName = "model-dir", doc = "Model directory")
    private GATKPath modelDir;

    @Argument(fullName = "device", doc = "Device for Torch backend (e.g. \"cpu\", \"cuda\")", optional = true)
    private String device = "cpu";

    @Argument(fullName = "random-seed", doc = "PRNG seed", optional = true)
    private int randomSeed = 92837488;

    @Argument(fullName = "predictive-samples", doc = "Number of samples for predictive distribution", optional = true)
    private int predictiveSamples = 1000;

    @Argument(fullName = "discrete-samples", doc = "Number of samples for discrete distribution", optional = true)
    private int discreteSamples = 1000;

    @Argument(fullName = "discrete-log-freq", doc = "Number of iterations between log messages for discrete sampling", optional = true)
    private int discreteLogFreq = 100;

    @Argument(fullName = "jit", doc = "Enable JIT compilation", optional = true)
    private boolean enableJit = false;

    @Hidden
    @Argument(fullName = "enable-journal", shortName = "enable-journal", doc = "Enable streaming process journal.", optional = true)
    private boolean enableJournal = false;

    @Hidden
    @Argument(fullName = "python-profile", shortName = "python-profile", doc = "Run the tool with the Python CProfiler on and write results to this file.", optional = true)
    private File pythonProfileResults;

    // Create the Python executor. This doesn't actually start the Python process, but verifies that
    // the requestedPython executable exists and can be located.
    final StreamingPythonScriptExecutor<String> pythonExecutor = new StreamingPythonScriptExecutor<>(true);

    private StructuralVariantType svType = null;
    private Map<String, VariantOutput> variantGenotypeDataMap = null;
    private List<String> sampleList;
    private VariantContextWriter vcfWriter;

    public static List<String> FORMAT_FIELDS = Lists.newArrayList(
            SVCluster.DISCORDANT_PAIR_COUNT_ATTRIBUTE,
            SVCluster.START_SPLIT_READ_COUNT_ATTRIBUTE,
            SVCluster.END_SPLIT_READ_COUNT_ATTRIBUTE,
            SVCopyNumberPosteriors.NEUTRAL_COPY_NUMBER_KEY,
            SVCopyNumberPosteriors.COPY_NUMBER_LOG_POSTERIORS_KEY
    );

    @Override
    public void onTraversalStart() {}

    @Override
    public void firstPassApply(final VariantContext variant,
                               final ReadsContext readsContext,
                               final ReferenceContext referenceContext,
                               final FeatureContext featureContext) {
        validateRecord(variant);
    }

    @Override
    public void afterFirstPass() {

        logger.info("Reading genotype model sample list...");
        final Path sampleListPath = Paths.get(modelDir.toString(), modelName + ".sample_ids.list");
        try (final BufferedReader file = new BufferedReader(IOUtils.makeReaderMaybeGzipped(sampleListPath))) {
            sampleList = file.lines().collect(Collectors.toList());
        } catch (final IOException e) {
            throw new RuntimeException("Error reading from genotype model samples list: "
                    + sampleListPath.toAbsolutePath().toString());
        }

        logger.info("Reading VCF sample list...");
        final List<String> vcfSampleList = getHeaderForVariants().getGenotypeSamples();
        if (!(sampleList.size() == vcfSampleList.size() && new HashSet<>(sampleList).containsAll(vcfSampleList))) {
            throw new UserException.BadInput("VCF and genotype model sample sets are not identical");
        }

        // Start the Python process and initialize a stream writer for streaming data to the Python code
        pythonExecutor.start(Collections.emptyList(), enableJournal, pythonProfileResults);
        pythonExecutor.initStreamWriter(AsynchronousStreamWriter.stringSerializer);

        // Execute Python code to initialize training
        logger.info("Sampling posterior distribution...");
        pythonExecutor.sendSynchronousCommand("import svgenotyper" + NL);
        pythonExecutor.sendSynchronousCommand("args = " + generatePythonArgumentsDictionary() + NL);
        final String runGenotypeCommand = "output, global_stats_by_type = svgenotyper.genotype.run(" +
                "args=args, svtype_str='" + svType.name() + "')" + NL;
        pythonExecutor.sendSynchronousCommand(runGenotypeCommand);
        logger.info("Sampling completed!");

        logger.info("Reading output file...");
        final Path intermediateFilePath = Paths.get(intermediatesDir.toString(), modelName + ".genotypes.tsv");
        try (final BufferedReader file = new BufferedReader(IOUtils.makeReaderMaybeGzipped(intermediateFilePath))) {
            final String header = file.readLine();
            if (!header.startsWith("#")) {
                throw new RuntimeException("Expected intermediate genotypes file header starting with #");
            }
            variantGenotypeDataMap = file.lines().map(line -> {
                final VariantOutput variantOutput = VariantOutput.ParseVariantOutput(line, sampleList);
                return new HashMap.SimpleImmutableEntry<>(variantOutput.id, variantOutput);
            }).collect(Collectors.toMap(HashMap.Entry::getKey, HashMap.Entry::getValue));
        } catch (final IOException e) {
            throw new RuntimeException("Error reading from intermediate genotypes file: "
                    + intermediateFilePath.toAbsolutePath().toString());
        }
        pythonExecutor.terminate();

        vcfWriter = createVCFWriter(outputVcf);

        final VCFHeader header = getHeaderForVariants();
        header.addMetaDataLine(VCFStandardHeaderLines.getFormatLine(VCFConstants.GENOTYPE_QUALITY_KEY, true));
        header.addMetaDataLine(VCFStandardHeaderLines.getFormatLine(VCFConstants.GENOTYPE_PL_KEY, true));
        header.addMetaDataLine(new VCFFormatHeaderLine(COPY_NUMBER_FIELD, 1, VCFHeaderLineType.Integer, "Copy number"));
        header.addMetaDataLine(new VCFInfoHeaderLine(PAIRED_END_PROB_FIELD, 1, VCFHeaderLineType.Float, "Paired-end read support probability"));
        header.addMetaDataLine(new VCFInfoHeaderLine(FIRST_SPLIT_READ_PROB_FIELD, 1, VCFHeaderLineType.Float, "First split read support probability"));
        header.addMetaDataLine(new VCFInfoHeaderLine(SECOND_SPLIT_READ_PROB_FIELD, 1, VCFHeaderLineType.Float, "Second read support probability"));
        header.addMetaDataLine(new VCFInfoHeaderLine(DEPTH_SUPPORT_PROB_FIELD, 1, VCFHeaderLineType.Float, "Read depth support probability"));
        header.addMetaDataLine(new VCFInfoHeaderLine(PAIRED_END_BACKGROUND_FIELD, 1, VCFHeaderLineType.Float, "Paired-end read mean background rate"));
        header.addMetaDataLine(new VCFInfoHeaderLine(FIRST_SPLIT_READ_BACKGROUND_FIELD, 1, VCFHeaderLineType.Float, "First split read mean background rate"));
        header.addMetaDataLine(new VCFInfoHeaderLine(SECOND_SPLIT_READ_BACKGROUND_FIELD, 1, VCFHeaderLineType.Float, "Second split read mean background rate"));
        header.addMetaDataLine(new VCFInfoHeaderLine(PAIRED_END_MEAN_BIAS_FIELD, 1, VCFHeaderLineType.Float, "Paired-end read mean bias"));
        header.addMetaDataLine(new VCFInfoHeaderLine(FIRST_SPLIT_READ_MEAN_BIAS_FIELD, 1, VCFHeaderLineType.Float, "First split read mean bias"));
        header.addMetaDataLine(new VCFInfoHeaderLine(SECOND_SPLIT_READ_MEAN_BIAS_FIELD, 1, VCFHeaderLineType.Float, "Second split read mean bias"));
        header.addMetaDataLine(new VCFInfoHeaderLine(HARDY_WEINBERG_Q_FIELD, 1, VCFHeaderLineType.Float, "Hardy-Weinberg q parameter"));
        header.addMetaDataLine(new VCFInfoHeaderLine(HARDY_WEINBERG_R_FIELD, 1, VCFHeaderLineType.Float, "Hardy-Weinberg r parameter"));
        vcfWriter.writeHeader(header);
    }

    @Override
    public void secondPassApply(final VariantContext variant,
                                final ReadsContext readsContext,
                                final ReferenceContext referenceContext,
                                final FeatureContext featureContext) {

        final String variantId = variant.getID();
        if (!variantGenotypeDataMap.containsKey(variantId)) {
            throw new UserException.BadInput("Could not find vcf record " + variantId + " in model output.");
        }
        final VariantOutput modelData = variantGenotypeDataMap.get(variant.getID());

        final VariantContextBuilder builder = new VariantContextBuilder(variant);

        builder.attribute(PAIRED_END_PROB_FIELD, modelData.getP_m_pe());
        builder.attribute(FIRST_SPLIT_READ_PROB_FIELD, modelData.getP_m_sr1());
        builder.attribute(SECOND_SPLIT_READ_PROB_FIELD, modelData.getP_m_sr2());
        builder.attribute(DEPTH_SUPPORT_PROB_FIELD, modelData.getP_m_rd());
        builder.attribute(PAIRED_END_BACKGROUND_FIELD, modelData.getEps_pe());
        builder.attribute(FIRST_SPLIT_READ_BACKGROUND_FIELD, modelData.getEps_sr1());
        builder.attribute(SECOND_SPLIT_READ_BACKGROUND_FIELD, modelData.getEps_sr2());
        builder.attribute(PAIRED_END_MEAN_BIAS_FIELD, modelData.getPhi_pe());
        builder.attribute(FIRST_SPLIT_READ_MEAN_BIAS_FIELD, modelData.getPhi_sr1());
        builder.attribute(SECOND_SPLIT_READ_MEAN_BIAS_FIELD, modelData.getPhi_sr2());
        builder.attribute(HARDY_WEINBERG_Q_FIELD, modelData.getEta_q());
        builder.attribute(HARDY_WEINBERG_R_FIELD, modelData.getEta_r());

        final StructuralVariantType svType = variant.getStructuralVariantType();
        final Allele refAllele = Allele.REF_N;
        final Allele altAllele = Allele.create("<" + svType.name() + ">", false);
        final AlleleList alleleList = new IndexedAlleleList(Lists.newArrayList(refAllele, altAllele));

        final int numGenotypes = modelData.getNumGenotypes();
        final int numSamples = sampleList.size();
        final int[] samplePloidies = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            samplePloidies[i] = getSamplePloidy(variant, sampleList.get(i));
        }

        final ArrayList<Genotype> newGenotypes = new ArrayList<>(variant.getNSamples());
        for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
            final String sample = sampleList.get(sampleIndex);
            final int samplePloidy = samplePloidies[sampleIndex];
            final Genotype genotype = variant.getGenotype(sample);
            final GenotypeBuilder genotypeBuilder = new GenotypeBuilder(genotype);

            final double[] freq = modelData.getSampleFrequencies(sampleIndex);
            final int[] genotypePhredProbs = new int[numGenotypes];
            for (int i = 0; i < numGenotypes; i++) {
                if (i <= samplePloidy || svType.equals(StructuralVariantType.DUP) || svType.equals(StructuralVariantType.BND)) {
                    genotypePhredProbs[i] = getGenotypePhredScore(freq[i]);
                } else {
                    throw new UserException.BadInput("Genotype state " + i + " not supported for SVTYPE " + svType.name()
                            + " when the sample ploidy is " + samplePloidy);
                }
            }
            genotypeBuilder.PL(genotypePhredProbs);

            double minPL = MAX_GENOTYPE_PL;
            int minPLIndex = -1;
            for (int i = 0; i < genotypePhredProbs.length; i++) {
                if (genotypePhredProbs[i] <= minPL) {
                    minPL = genotypePhredProbs[i];
                    minPLIndex = i;
                }
            }

            double secondMinPL = MAX_GENOTYPE_PL;
            for (int i = 0; i < genotypePhredProbs.length; i++) {
                if (i != minPLIndex) {
                    if (genotypePhredProbs[i] < secondMinPL) {
                        secondMinPL = genotypePhredProbs[i];
                    }
                }
            }
            final int GQ = (int) Math.round(secondMinPL - minPL);
            genotypeBuilder.GQ(GQ);

            // TODO: multi-allelic sites, allosomes
            final int copyNumber;
            final List<Allele> genotypeAlleles;
            if (svType.equals(StructuralVariantType.DEL)
                    || svType.equals(StructuralVariantType.INS)
                    || svType.equals(StructuralVariantType.BND)) {
                final int numRefAlleles = samplePloidy - minPLIndex;
                final int numAltAlleles = minPLIndex;
                genotypeAlleles = new ArrayList<>(samplePloidy);
                for (int i = 0; i < numRefAlleles; i++) {
                    genotypeAlleles.add(refAllele);
                }
                for (int i = 0; i < numAltAlleles; i++) {
                    genotypeAlleles.add(altAllele);
                }
                copyNumber = numRefAlleles;
            } else if (svType.equals(StructuralVariantType.DUP)) {
                genotypeAlleles = new ArrayList(samplePloidy);
                for (int i = 0; i < samplePloidy; i++) {
                    genotypeAlleles.add(Allele.NO_CALL);
                }
                copyNumber = samplePloidy + minPLIndex;
            } else {
                throw new UserException.BadInput("Unsupported SVTYPE: " + svType);
            }
            genotypeBuilder.alleles(genotypeAlleles);
            genotypeBuilder.attribute(COPY_NUMBER_FIELD, copyNumber);
            newGenotypes.add(genotypeBuilder.make());
        }
        builder.genotypes(newGenotypes);

        //TODO: using default HC args
        //final GenotypeCalculationArgumentCollection config = new HaplotypeCallerArgumentCollection().standardArgs.genotypeArgs;
        //final AlleleFrequencyCalculator afCalculator = AlleleFrequencyCalculator.makeCalculator(config);
        // TODO: account for ploidy
        //final double log10Confidence = afCalculator.calculate(builder.make(), 2).log10ProbVariantPresent();
        double log10ProbNoVariant = 0;
        for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
            final double[] freq = modelData.getSampleFrequencies(sampleIndex);
            if (freq.length > 0) {
                log10ProbNoVariant += FastMath.log10(freq[0]);
            }
        }
        log10ProbNoVariant = Math.max(MIN_SITE_LOG10_PROB_NO_VAR, log10ProbNoVariant);
        builder.log10PError(log10ProbNoVariant);
        vcfWriter.add(builder.make());
    }

    private static int getGenotypePhredScore(final double prob) {
        return prob <= MIN_GENOTYPE_PROB ? MAX_GENOTYPE_PL : (int) (-10. * FastMath.log10(prob));
    }

    private int getSamplePloidy(final VariantContext variant, final String sample) {
        Utils.validateArg(variant.hasGenotype(sample), "Genotype not found for sample: " + sample);
        final Genotype genotype = variant.getGenotype(sample);
        Utils.validateArg(genotype.hasExtendedAttribute(SVCopyNumberPosteriors.NEUTRAL_COPY_NUMBER_KEY),
                "Variant " + variant.getID() + " missing format field " + SVCopyNumberPosteriors.NEUTRAL_COPY_NUMBER_KEY
                        + " for sample " + sample);
        return Integer.valueOf(((String)genotype.getExtendedAttribute(SVCopyNumberPosteriors.NEUTRAL_COPY_NUMBER_KEY)));
    }

    private final static class SVAlleleEvidence implements Locatable {

        final SimpleInterval interval;

        public SVAlleleEvidence(final SimpleInterval interval) {
            this.interval = interval;
        }

        @Override
        public String getContig() {
            return interval.getContig();
        }

        @Override
        public int getStart() {
            return interval.getStart();
        }

        @Override
        public int getEnd() {
            return interval.getEnd();
        }
    }

    private void validateRecord(final VariantContext variant) {
        if (svType == null) {
            svType = variant.getStructuralVariantType();
        } else {
            if (!variant.getStructuralVariantType().equals(svType)) {
                throw new UserException.BadInput("Variants must all have the same SVTYPE. First variant was "
                        + svType.name() + " but found " + variant.getStructuralVariantType().name() + " for record " + variant.getID());
            }
        }
    }

    @Override
    public Object onTraversalSuccess() {
        vcfWriter.close();
        return null;
    }

    private String generatePythonArgumentsDictionary() {
        final List<String> arguments = new ArrayList<>();
        arguments.add("'output_dir': '" + intermediatesDir + "'");
        arguments.add("'model_name': '" + modelName + "'");
        arguments.add("'model_dir': '" + modelDir + "'");
        arguments.add("'device': '" + device + "'");
        arguments.add("'random_seed': " + randomSeed);
        arguments.add("'genotype_predictive_samples': " + predictiveSamples);
        arguments.add("'genotype_discrete_samples': " + discreteSamples);
        arguments.add("'genotype_discrete_log_freq': " + discreteLogFreq);
        arguments.add("'jit': " + (enableJit ? "True" : "False"));
        return "{ " + String.join(", ", arguments) + " }";
    }

    private static final class VariantOutput {
        private final List<String> sampleList;
        private final String id;
        private final List<double[]> frequencies;
        private final double p_m_pe;
        private final double p_m_sr1;
        private final double p_m_sr2;
        private final double p_m_rd;
        private final double eps_pe;
        private final double eps_sr1;
        private final double eps_sr2;
        private final double phi_pe;
        private final double phi_sr1;
        private final double phi_sr2;
        private final double eta_q;
        private final double eta_r;

        public VariantOutput(final List<String> sampleList,
                             final String id,
                             final List<double[]> frequencies,
                             final double p_m_pe,
                             final double p_m_sr1,
                             final double p_m_sr2,
                             final double p_m_rd,
                             final double eps_pe,
                             final double eps_sr1,
                             final double eps_sr2,
                             final double phi_pe,
                             final double phi_sr1,
                             final double phi_sr2,
                             final double eta_q,
                             final double eta_r) {
            this.sampleList = sampleList;
            this.id = id;
            this.frequencies = frequencies;
            this.p_m_pe = p_m_pe;
            this.p_m_sr1 = p_m_sr1;
            this.p_m_sr2 = p_m_sr2;
            this.p_m_rd = p_m_rd;
            this.eps_pe = eps_pe;
            this.eps_sr1 = eps_sr1;
            this.eps_sr2 = eps_sr2;
            this.phi_pe = phi_pe;
            this.phi_sr1 = phi_sr1;
            this.phi_sr2 = phi_sr2;
            this.eta_q = eta_q;
            this.eta_r = eta_r;
        }

        public static VariantOutput ParseVariantOutput(final String line, final List<String> sampleList) {
            final String[] values = line.trim().split(COLUMN_SEPARATOR);
            final String id = values[0];
            final String[] freqStringArray = values[1].split(FIRST_DIM_SEPARATOR);
            final int numSamples = sampleList.size();
            if (freqStringArray.length != numSamples) {
                throw new UserException.BadInput("Genotype frequencies did not match sample list size");
            }
            final List<double[]> freqList = new ArrayList<>(freqStringArray.length);
            for (int i = 0; i < numSamples; i++) {
                final String[] sampleFreqStringArray = freqStringArray[i].split(SECOND_DIM_SEPARATOR);
                final double[] sampleFreq = new double[sampleFreqStringArray.length];
                for (int j = 0; j < sampleFreq.length; j++) {
                    sampleFreq[j] = Double.parseDouble(sampleFreqStringArray[j]);
                }
                freqList.add(sampleFreq);
            }

            final double p_m_pe = Double.parseDouble(values[2]);
            final double p_m_sr1 = Double.parseDouble(values[3]);
            final double p_m_sr2 = Double.parseDouble(values[4]);
            final double p_m_rd = Double.parseDouble(values[5]);
            final double eps_pe = Double.parseDouble(values[6]);
            final double eps_sr1 = Double.parseDouble(values[7]);
            final double eps_sr2 = Double.parseDouble(values[8]);
            final double phi_pe = Double.parseDouble(values[9]);
            final double phi_sr1 = Double.parseDouble(values[10]);
            final double phi_sr2 = Double.parseDouble(values[11]);
            final double eta_q = Double.parseDouble(values[12]);
            final double eta_r = Double.parseDouble(values[13]);

            return new VariantOutput(
                    sampleList,
                    id,
                    freqList,
                    p_m_pe,
                    p_m_sr1,
                    p_m_sr2,
                    p_m_rd,
                    eps_pe,
                    eps_sr1,
                    eps_sr2,
                    phi_pe,
                    phi_sr1,
                    phi_sr2,
                    eta_q,
                    eta_r
            );
        }

        public int getNumGenotypes() {
            if (frequencies.isEmpty()) return 0;
            final int max = frequencies.stream().mapToInt(f -> f.length).max().getAsInt();
            final int min = frequencies.stream().mapToInt(f -> f.length).min().getAsInt();
            if (max != min) {
                throw new UserException.BadInput("Genotype frequency arrays not of uniform size");
            }
            return max;
        }

        public String getId() {
            return id;
        }

        public double[] getSampleFrequencies(final int sampleIndex) {
            Utils.validateArg(sampleIndex >= 0 && sampleIndex < sampleList.size(), "Invalid sample index: " + sampleIndex);
            return frequencies.get(sampleIndex);
        }

        public double getP_m_pe() {
            return p_m_pe;
        }

        public double getP_m_sr1() {
            return p_m_sr1;
        }

        public double getP_m_sr2() {
            return p_m_sr2;
        }

        public double getP_m_rd() {
            return p_m_rd;
        }

        public double getEps_pe() {
            return eps_pe;
        }

        public double getEps_sr1() {
            return eps_sr1;
        }

        public double getEps_sr2() {
            return eps_sr2;
        }

        public double getPhi_pe() {
            return phi_pe;
        }

        public double getPhi_sr1() {
            return phi_sr1;
        }

        public double getPhi_sr2() {
            return phi_sr2;
        }

        public double getEta_q() {
            return eta_q;
        }

        public double getEta_r() {
            return eta_r;
        }
    }
}
