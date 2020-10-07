package org.broadinstitute.hellbender.tools.sv;

import com.google.common.collect.Lists;
import htsjdk.variant.variantcontext.Genotype;
import htsjdk.variant.variantcontext.StructuralVariantType;
import htsjdk.variant.variantcontext.VariantContext;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.barclay.argparser.Hidden;
import org.broadinstitute.barclay.help.DocumentedFeature;
import org.broadinstitute.hellbender.engine.FeatureContext;
import org.broadinstitute.hellbender.engine.MultiplePassVariantWalker;
import org.broadinstitute.hellbender.engine.ReadsContext;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.utils.io.IOUtils;
import org.broadinstitute.hellbender.utils.python.StreamingPythonScriptExecutor;
import org.broadinstitute.hellbender.utils.runtime.AsynchronousStreamWriter;
import picard.cmdline.programgroups.VariantFilteringProgramGroup;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
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
        programGroup = VariantFilteringProgramGroup.class
)

public class SVTrainGenotyping extends MultiplePassVariantWalker {

    private final static String NL = String.format("%n");
    private static final String DATA_VALUE_SEPARATOR = ";";
    private static final String DATA_TYPE_SEPARATOR = "\t";

    static final String USAGE_ONE_LINE_SUMMARY = "Train model to genotype structural variants";
    static final String USAGE_SUMMARY = "Runs training on a set of variants and generates a genotyping model.";

    @Argument(fullName = "coverage-file", doc = "Tab-delimited table of sample mean coverage")
    private File coverageFile;

    @Argument(fullName = "output-name", doc = "Output name")
    private String outputName;

    @Argument(fullName = "output-dir", doc = "Output directory")
    private String outputDir;

    @Argument(fullName = "device", doc = "Device for Torch backend (e.g. \"cpu\", \"cuda\")", optional = true)
    private String device = "cpu";

    @Argument(fullName = "random-seed", doc = "PRNG seed", optional = true)
    private int randomSeed = 92837488;

    @Argument(fullName = "num-states", doc = "Max number of genotype states (0 for auto)", optional = true)
    private int numStates = 0;

    @Argument(fullName = "depth-dilution-factor", doc = "Dilution factor for depth posteriors", optional = true)
    private double depthDilutionFactor = 1e-6;

    @Argument(fullName = "eps-pe", doc = "Mean of PE noise prior", optional = true)
    private double epsilonPE  = 0.1;

    @Argument(fullName = "eps-sr1", doc = "Mean of SR1 noise prior", optional = true)
    private double epsilonSR1  = 0.1;

    @Argument(fullName = "eps-sr2", doc = "Mean of SR2 noise prior", optional = true)
    private double epsilonSR2  = 0.1;

    @Argument(fullName = "lambda-pe", doc = "Mean of PE variance factor prior", optional = true)
    private double lambdaPE  = 0.1;

    @Argument(fullName = "lambda-sr1", doc = "Mean of SR1 variance factor prior", optional = true)
    private double lambdaSR1  = 0.1;

    @Argument(fullName = "lambda-sr2", doc = "Mean of SR2 variance factor prior", optional = true)
    private double lambdaSR2  = 0.1;

    @Argument(fullName = "phi-pe", doc = "Variance of PE bias prior", optional = true)
    private double phiPE  = 0.1;

    @Argument(fullName = "phi-sr1", doc = "Variance of SR1 bias prior", optional = true)
    private double phiSR1  = 0.1;

    @Argument(fullName = "phi-sr2", doc = "Variance of SR2 bias prior", optional = true)
    private double phiSR2  = 0.1;

    @Argument(fullName = "eta-r", doc = "Mean of allele frequency prior", optional = true)
    private double etaQ  = 0.1;

    @Argument(fullName = "eta-q", doc = "Mean of secondary allele prior (DUP only)", optional = true)
    private double etaR  = 0.01;

    @Argument(fullName = "lr-decay", doc = "Learning rate decay constant (lower is faster)", optional = true)
    private double lrDecay = 1000;

    @Argument(fullName = "lr-min", doc = "Minimum learning rate", optional = true)
    private double lrMin = 1e-3;

    @Argument(fullName = "lr-init", doc = "Initial learning rate", optional = true)
    private double lrInit = 0.01;

    @Argument(fullName = "adam-beta1", doc = "ADAM beta1 constant", optional = true)
    private double adamBeta1 = 0.9;

    @Argument(fullName = "adam-beta2", doc = "ADAM beta2 constant", optional = true)
    private double adamBeta2 = 0.999;

    @Argument(fullName = "max-iter", doc = "Max number of training iterations", optional = true)
    private int maxIter = 2000;

    @Argument(fullName = "iter-log-freq", doc ="Number of iterations between log messages", optional = true)
    private int iterLogFreq = 100;

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

    private File samplesFile = null;
    private List<String> batchList = new ArrayList<>();

    public static List<StructuralVariantType> SV_TYPES = Lists.newArrayList(
            StructuralVariantType.DEL,
            StructuralVariantType.DUP,
            StructuralVariantType.INS,
            StructuralVariantType.INV,
            StructuralVariantType.BND
    );

    public static List<String> FORMAT_FIELDS = Lists.newArrayList(
            SVCluster.DISCORDANT_PAIR_COUNT_ATTRIBUTE,
            SVCluster.START_SPLIT_READ_COUNT_ATTRIBUTE,
            SVCluster.END_SPLIT_READ_COUNT_ATTRIBUTE,
            SVCopyNumberPosteriors.NEUTRAL_COPY_NUMBER_KEY,
            SVCopyNumberPosteriors.COPY_NUMBER_LOG_POSTERIORS_KEY
    );

    @Override
    protected int numberOfPasses() {
        return SV_TYPES.size();
    }

    @Override
    public void onTraversalStart() {
        samplesFile = createSampleList();

        // Start the Python process and initialize a stream writer for streaming data to the Python code
        pythonExecutor.start(Collections.emptyList(), enableJournal, pythonProfileResults);
        pythonExecutor.initStreamWriter(AsynchronousStreamWriter.stringSerializer);

        // Execute Python code to initialize training
        pythonExecutor.sendSynchronousCommand("import svgenotyper" + NL);
        pythonExecutor.sendSynchronousCommand("args = " + generatePythonArgumentsDictionary() + NL);
    }

    @Override
    protected void nthPassApply(final VariantContext variant,
                             final ReadsContext readsContext,
                             final ReferenceContext referenceContext,
                             final FeatureContext featureContext,
                             final int n) {
        final StructuralVariantType svType = variant.getStructuralVariantType();
        if (svType.equals(SV_TYPES.get(n)) && !isDepthOnly(variant)) {
            transferToPythonViaFifo(variant);
        }
    }

    @Override
    protected void afterNthPass(final int n) {
        final int batchSize = batchList.size();
        if (batchSize > 0) {
            final String svType = SV_TYPES.get(n).name();
            final String pythonCommand = String.format("svgenotyper.train.run(args=args, batch_size=%d, svtype_str='%s')",
                    batchSize, svType) + NL;
            pythonExecutor.startBatchWrite(pythonCommand, batchList);
            pythonExecutor.waitForPreviousBatchCompletion();
            batchList = new ArrayList<>();
        }
    }

    @Override
    public Object onTraversalSuccess() {
        pythonExecutor.terminate();
        return null;
    }

    private void transferToPythonViaFifo(final VariantContext variant) {
        final Map<String, StringBuilder> stringBuilderMap = FORMAT_FIELDS.stream()
                .collect(Collectors.toMap(s -> s, s -> new StringBuilder()));
        final int numGenotypes = variant.getNSamples();
        int genotypeIndex = 0;
        for (final Genotype genotype : variant.getGenotypes()) {
            final String separator = genotypeIndex == numGenotypes - 1 ? "" : DATA_VALUE_SEPARATOR;
            for (final Map.Entry<String, StringBuilder> entry : stringBuilderMap.entrySet()) {
                entry.getValue().append(genotype.getExtendedAttribute(entry.getKey()) + separator);
            }
            genotypeIndex++;
        }
        final StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(variant.getID() + DATA_TYPE_SEPARATOR);
        final int numFields = FORMAT_FIELDS.size();
        int attributeIndex = 0;
        for (final String attribute : FORMAT_FIELDS) {
            final String separator = attributeIndex == numFields - 1 ? "" : DATA_TYPE_SEPARATOR;
            stringBuilder.append(stringBuilderMap.get(attribute).toString() + separator);
            attributeIndex++;
        }
        batchList.add(stringBuilder.toString());
    }

    private boolean isDepthOnly(final VariantContext variant) {
        return variant.getAttributeAsString(SVCluster.ALGORITHMS_ATTRIBUTE, "").equals(SVCluster.DEPTH_ALGORITHM);
    }

    private File createSampleList() {
        final List<String> samples = getHeaderForVariants().getSampleNamesInOrder();
        return IOUtils.writeTempFile(samples, outputName + ".samples", ".tmp");
    }

    private String generatePythonArgumentsDictionary() {
        if (samplesFile == null) {
            throw new RuntimeException("Samples file doesn't exist");
        }
        final List<String> arguments = new ArrayList<>();
        arguments.add("'coverage_file': '" + coverageFile.getAbsolutePath() + "'");
        arguments.add("'samples_file': '" + samplesFile.getAbsolutePath() + "'");
        arguments.add("'output_name': '" + outputName + "'");
        arguments.add("'output_dir': '" + outputDir + "'");
        arguments.add("'device': '" + device + "'");
        arguments.add("'num_states': " + (numStates == 0 ? "None" : numStates));
        arguments.add("'random_seed': " + randomSeed);
        if (numStates != 0) {
            arguments.add("'num_states': " + numStates);
        }
        arguments.add("'depth_dilution_factor': " + depthDilutionFactor);
        arguments.add("'eps_pe': " + epsilonPE);
        arguments.add("'eps_sr1': " + epsilonSR1);
        arguments.add("'eps_sr2': " + epsilonSR2);
        arguments.add("'lambda_pe': " + lambdaPE);
        arguments.add("'lambda_sr1': " + lambdaSR1);
        arguments.add("'lambda_sr2': " + lambdaSR2);
        arguments.add("'phi_pe': " + phiPE);
        arguments.add("'phi_sr1': " + phiSR1);
        arguments.add("'phi_sr2': " + phiSR2);
        arguments.add("'eta_q': " + etaQ);
        arguments.add("'eta_r': " + etaR);
        arguments.add("'lr_decay': " + lrDecay);
        arguments.add("'lr_min': " + lrMin);
        arguments.add("'lr_init': " + lrInit);
        arguments.add("'adam_beta1': " + adamBeta1);
        arguments.add("'adam_beta2': " + adamBeta2);
        arguments.add("'max_iter': " + maxIter);
        arguments.add("'iter_log_freq': " + iterLogFreq);
        if (enableJit) {
            arguments.add("'jit': True");
        }
        return "{ " + String.join(", ", arguments) + " }";
    }
}
