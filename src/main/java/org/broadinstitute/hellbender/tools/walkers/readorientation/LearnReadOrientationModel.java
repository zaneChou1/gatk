package org.broadinstitute.hellbender.tools.walkers.readorientation;

import com.google.common.annotations.VisibleForTesting;
import htsjdk.samtools.metrics.MetricsFile;
import htsjdk.samtools.util.*;
import org.apache.commons.lang3.tuple.Pair;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.hellbender.cmdline.CommandLineProgram;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.cmdline.programgroups.ShortVariantDiscoveryProgramGroup;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.tools.walkers.mutect.M2ArgumentCollection;
import org.broadinstitute.hellbender.utils.Nucleotide;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.io.IOUtils;
import org.broadinstitute.hellbender.tools.walkers.mutect.Mutect2;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Learn the prior probability of read orientation artifact from the output of {@link CollectF1R2Counts} of {@link Mutect2}
 * Details of the model may be found in docs/mutect/mutect.pdf.
 *
 *
 * <h3>Usage Examples</h3>
 *
 * gatk LearnReadOrientationModel \
 *   -I f1r2.tar.gz \
 *   -O artifact-prior.tar.gz
 *
 *   Note that the -I argument may be specified multiple times, as in the case of learning a model from multiple scatters of the same sample.
 *   The input F1R2 tar.gz counts can be generated by CollectF1R2Counts or by Mutect2, with the --f1r2-tar-gz argument.  If the inputs contains
 *   F1R2 counts for multiple samples then the output file contains learned artifact priors for the same samples.
 */
@CommandLineProgramProperties(
        summary = "Get the maximum likelihood estimates of artifact prior probabilities in the orientation bias mixture model filter",
        oneLineSummary = "Get the maximum likelihood estimates of artifact prior probabilities in the orientation bias mixture model filter",
        programGroup = ShortVariantDiscoveryProgramGroup.class
)
public class LearnReadOrientationModel extends CommandLineProgram {
    public static final double DEFAULT_CONVERGENCE_THRESHOLD = 1e-4;
    public static final int DEFAULT_MAX_ITERATIONS = 20;
    private static final int DEFAULT_INITIAL_LIST_SIZE = 1_000_000;

    public static final String EM_CONVERGENCE_THRESHOLD_LONG_NAME = "convergence-threshold";
    public static final String MAX_EM_ITERATIONS_LONG_NAME = "num-em-iterations";
    public static final String MAX_DEPTH_LONG_NAME = "max-depth";
    public static final String ARTIFACT_PRIOR_EXTENSION = ".orientation_priors";

    @Argument(fullName = StandardArgumentDefinitions.INPUT_LONG_NAME, shortName = StandardArgumentDefinitions.INPUT_SHORT_NAME,
            doc = "One or more .tar.gz containing outputs of CollectF1R2Counts")
    private List<File> inputTarGzs;

    @Argument(fullName = StandardArgumentDefinitions.OUTPUT_LONG_NAME, shortName = StandardArgumentDefinitions.OUTPUT_SHORT_NAME, doc = "tar.gz of artifact prior tables")
    private File outputTarGz;

    @Argument(fullName = EM_CONVERGENCE_THRESHOLD_LONG_NAME, doc = "Stop the EM when the distance between parameters between iterations falls below this value", optional = true)
    private double convergenceThreshold = DEFAULT_CONVERGENCE_THRESHOLD;

    @Argument(fullName = MAX_EM_ITERATIONS_LONG_NAME, doc = "give up on EM after this many iterations", optional = true)
    private int maxEMIterations = DEFAULT_MAX_ITERATIONS;

    @Argument(fullName = MAX_DEPTH_LONG_NAME, doc = "sites with depth higher than this value will be grouped", optional = true)
    private int maxDepth = F1R2FilterConstants.DEFAULT_MAX_DEPTH;

    private Map<String, List<Histogram<Integer>>> refHistogramsBySample;

    private Map<String, List<Histogram<Integer>>> altHistogramsBySample;

    @Override
    public Object doWork(){
        if (!outputTarGz.getAbsolutePath().endsWith(".tar.gz")) {
            throw new UserException.CouldNotCreateOutputFile(outputTarGz,  "Output file must end in .tar.gz");
        }

        final List<File> tmpDirs = IntStream.range(0, inputTarGzs.size())
                .mapToObj(n -> IOUtils.createTempDir(Integer.toString(n)))
                .collect(Collectors.toList());

        IntStream.range(0, inputTarGzs.size()).forEach(n -> IOUtils.extractTarGz(inputTarGzs.get(n).toPath(), tmpDirs.get(n).toPath()));

        final List<File> refHistogramFiles = tmpDirs.stream().flatMap(dir -> F1R2CountsCollector.getRefHistogramsFromExtractedTar(dir).stream()).collect(Collectors.toList());
        final List<File> altHistogramFiles = tmpDirs.stream().flatMap(dir -> F1R2CountsCollector.getAltHistogramsFromExtractedTar(dir).stream()).collect(Collectors.toList());
        final List<File> altTableFiles = tmpDirs.stream().flatMap(dir -> F1R2CountsCollector.getAltTablesFromExtractedTar(dir).stream()).collect(Collectors.toList());

        // TODO: this is brittle: it relies on the fact that in CollectF1R2Counts we put a single header line with the same name in the ref and alt histograms
        final Map<String, List<MetricsFile<?, Integer>>> refHistogramMetricsFilesBySample = refHistogramFiles.stream()
                .map(file -> readMetricsFile(file))
                .collect(Collectors.groupingBy(metricsFile -> metricsFile.getHeaders().get(0).toString()));

        final Map<String, List<MetricsFile<?, Integer>>> altHistogramMetricsFilesBySample = altHistogramFiles.stream()
                .map(file -> readMetricsFile(file))
                .collect(Collectors.groupingBy(metricsFile -> metricsFile.getHeaders().get(0).toString()));

        final Set<String> refHistogramSamples = refHistogramMetricsFilesBySample.keySet();
        final Set<String> altHistogramSamples = altHistogramMetricsFilesBySample.keySet();
        Utils.validate(altHistogramSamples.isEmpty() || refHistogramSamples.containsAll(altHistogramSamples) && altHistogramSamples.containsAll(refHistogramSamples), "ref and alt histograms must have same samples");
        Utils.validate(altHistogramSamples.isEmpty() || refHistogramSamples.stream().allMatch(sample -> refHistogramMetricsFilesBySample.get(sample).size() == altHistogramMetricsFilesBySample.get(sample).size()),
                "Each sample must have the same number of alt and ref histograms");

        //Utils.validate(altHistogramFiles.size() == 0 || altDataTables.size() == altHistogramFiles.size(), "If provided, the number of alt histograms must match others");
        refHistogramsBySample = refHistogramSamples.stream().collect(Collectors.toMap(sample -> sample,
                sample -> sumHistogramsFromFiles(refHistogramMetricsFilesBySample.get(sample), true)));

        altHistogramsBySample = altHistogramSamples.stream().collect(Collectors.toMap(sample -> sample,
                sample -> sumHistogramsFromFiles(altHistogramMetricsFilesBySample.get(sample), false)));

        final Map<String, List<AltSiteRecord>> recordsBySample = gatherAltSiteRecords(altTableFiles);

        final Map<String, ArtifactPriorCollection> artifactPriorCollectionBySample = new LinkedHashMap<>();
        for (final Map.Entry<String, List<AltSiteRecord>> entry : recordsBySample.entrySet()) {
            final String sample = entry.getKey();
            final List<AltSiteRecord> records = entry.getValue();

            final Map<String, List<AltSiteRecord>> altDesignMatrixByContext = records.stream()
                    .collect(Collectors.groupingBy(AltSiteRecord::getReferenceContext));

            final ArtifactPriorCollection artifactPriorCollection = new ArtifactPriorCollection(sample);

            // Since e.g. G->T under AGT F1R2 is equivalent to C->A under ACT F2R1, combine the data
            for (final String refContext : F1R2FilterConstants.CANONICAL_KMERS) {
                final String reverseComplement = SequenceUtil.reverseComplement(refContext);

                // Merge ref histograms
                final Histogram<Integer> refHistogram = refHistogramsBySample.get(sample).stream()
                        .filter(h -> h.getValueLabel().equals(refContext))
                        .findFirst().orElseGet(() -> F1R2FilterUtils.createRefHistogram(refContext, maxDepth));
                final Histogram<Integer> refHistogramRevComp = refHistogramsBySample.get(sample).stream()
                        .filter(h -> h.getValueLabel().equals(reverseComplement))
                        .findFirst().orElseGet(() -> F1R2FilterUtils.createRefHistogram(reverseComplement, maxDepth));
                final Histogram<Integer> combinedRefHistograms = combineRefHistogramWithRC(refContext, refHistogram, refHistogramRevComp, maxDepth);


                // Merge alt depth=1 histograms
                final List<Histogram<Integer>> altDepthOneHistogramsForContext = !altHistogramsBySample.containsKey(sample) ? Collections.emptyList() :
                        altHistogramsBySample.get(sample).stream()
                        .filter(h -> h.getValueLabel().startsWith(refContext))
                        .collect(Collectors.toList());
                final List<Histogram<Integer>> altDepthOneHistogramsRevComp = !altHistogramsBySample.containsKey(sample) ? Collections.emptyList() :
                        altHistogramsBySample.get(sample).stream()
                        .filter(h -> h.getValueLabel().startsWith(reverseComplement))
                        .collect(Collectors.toList());
                final List<Histogram<Integer>> combinedAltHistograms = combineAltDepthOneHistogramWithRC(altDepthOneHistogramsForContext, altDepthOneHistogramsRevComp, maxDepth);

                // Finally, merge the rest of alt records
                final List<AltSiteRecord> altDesignMatrix = altDesignMatrixByContext.getOrDefault(refContext, new ArrayList<>()); // Cannot use Collections.emptyList() here because the input list must be mutable
                final List<AltSiteRecord> altDesignMatrixRevComp = altDesignMatrixByContext.getOrDefault(reverseComplement, Collections.emptyList());
                // Warning: the below method will mutate the content of {@link altDesignMatrixRevComp} and append to {@code altDesignMatrix}
                mergeDesignMatrices(altDesignMatrix, altDesignMatrixRevComp);


                if (combinedRefHistograms.getSumOfValues() == 0 || altDesignMatrix.isEmpty()) {
                    logger.info(String.format("Skipping the reference context %s as we didn't find either the ref or alt table for the context", refContext));
                    continue;
                }

                final LearnReadOrientationModelEngine engine = new LearnReadOrientationModelEngine(
                        combinedRefHistograms,
                        combinedAltHistograms,
                        altDesignMatrix,
                        convergenceThreshold,
                        maxEMIterations,
                        maxDepth,
                        logger);
                final ArtifactPrior artifactPrior = engine.learnPriorForArtifactStates();
                artifactPriorCollection.set(artifactPrior);
            }
            artifactPriorCollectionBySample.put(sample, artifactPriorCollection);
        }

        final File tmpPriorDir = IOUtils.createTempDir("priors");
        for (final String sample : artifactPriorCollectionBySample.keySet()) {
            final ArtifactPriorCollection artifactPriorCollection = artifactPriorCollectionBySample.get(sample);
            final File destination = new File(tmpPriorDir, IOUtils.urlEncode(sample) + ARTIFACT_PRIOR_EXTENSION);
            artifactPriorCollection.writeArtifactPriors(destination);
        }

        try {
            IOUtils.writeTarGz(outputTarGz.getAbsolutePath(), tmpPriorDir.listFiles());
        } catch (IOException ex) {
            throw new UserException.CouldNotCreateOutputFile("Could not create output .tar.gz file.", ex);
        }


        return "SUCCESS";
    }

    @VisibleForTesting
    public static Histogram<Integer> combineRefHistogramWithRC(final String refContext,
                                                               final Histogram<Integer> refHistogram,
                                                               final Histogram<Integer> refHistogramRevComp,
                                                               final int maxDepth){
        Utils.validateArg(refHistogram.getValueLabel()
                .equals(SequenceUtil.reverseComplement(refHistogramRevComp.getValueLabel())),
                "ref context = " + refHistogram.getValueLabel() + ", rev comp = " + refHistogramRevComp.getValueLabel());
        Utils.validateArg(refHistogram.getValueLabel().equals(refContext), "this better match");

        final Histogram<Integer> combinedRefHistogram = F1R2FilterUtils.createRefHistogram(refContext, maxDepth);

        for (final Integer depth : refHistogram.keySet()){
            final double newCount = refHistogram.get(depth).getValue() + refHistogramRevComp.get(depth).getValue();
            combinedRefHistogram.increment(depth, newCount);
        }

        return combinedRefHistogram;
    }

    @VisibleForTesting
    public static List<Histogram<Integer>> combineAltDepthOneHistogramWithRC(final List<Histogram<Integer>> altHistograms,
                                                                             final List<Histogram<Integer>> altHistogramsRevComp,
                                                                             final int maxDepth){
        if (altHistograms.isEmpty() && altHistogramsRevComp.isEmpty()){
            return Collections.emptyList();
        }

        final String refContext = ! altHistograms.isEmpty() ?
                F1R2FilterUtils.labelToTriplet(altHistograms.get(0).getValueLabel()).getLeft() :
                SequenceUtil.reverseComplement(F1R2FilterUtils.labelToTriplet(altHistogramsRevComp.get(0).getValueLabel()).getLeft());

        // Contract: altHistogram must be of the canonical representation of the kmer
        Utils.validateArg(F1R2FilterConstants.CANONICAL_KMERS.contains(refContext), "refContext must be the canonical representation but got " + refContext);

        final List<Histogram<Integer>> combinedHistograms = new ArrayList<>(F1R2FilterConstants.numAltHistogramsPerContext);

        for (Nucleotide altAllele : Nucleotide.STANDARD_BASES){
            // Skip when the alt base is the ref base, which doesn't make sense because this is a histogram of alt sites
            if (altAllele == F1R2FilterUtils.getMiddleBase(refContext)){
                continue;
            }

            final String reverseComplement = SequenceUtil.reverseComplement(refContext);
            final Nucleotide altAlleleRevComp = Nucleotide.valueOf(SequenceUtil.reverseComplement(altAllele.toString()));

            for (ReadOrientation orientation : ReadOrientation.values()) {
                final ReadOrientation otherOrientation = ReadOrientation.getOtherOrientation(orientation);
                final Histogram<Integer> altHistogram = altHistograms.stream()
                        .filter(h -> h.getValueLabel().equals(F1R2FilterUtils.tripletToLabel(refContext, altAllele, orientation)))
                        .findFirst().orElseGet(() -> F1R2FilterUtils.createAltHistogram(refContext, altAllele, orientation, maxDepth));

                final Histogram<Integer> altHistogramRevComp = altHistogramsRevComp.stream()
                        .filter(h -> h.getValueLabel().equals(F1R2FilterUtils.tripletToLabel(reverseComplement, altAlleleRevComp, otherOrientation)))
                        .findFirst().orElseGet(() -> F1R2FilterUtils.createAltHistogram(reverseComplement, altAlleleRevComp, otherOrientation, maxDepth));

                final Histogram<Integer> combinedHistogram = F1R2FilterUtils.createAltHistogram(refContext, altAllele, orientation, maxDepth);

                // Add the histograms manually - I don't like the addHistogram() in htsjdk method because it does so with side-effect
                for (final Integer depth : altHistogram.keySet()){
                    final double newCount = altHistogram.get(depth).getValue() + altHistogramRevComp.get(depth).getValue();
                    combinedHistogram.increment(depth, newCount);
                }
                combinedHistograms.add(combinedHistogram);
            }
        }

        return combinedHistograms;
    }


    /**
     * Contract: this method must be called after grouping the design matrices by context.
     * That is, {@param altDesignMatrix} must be a list of {@link AltSiteRecord} of a single reference context
     * (which is in F1R2FilterConstants.CANONICAL_KMERS) and {@param altDesignRevComp} contains only
     * {@link AltSiteRecord} of its reverse complement.
     */
    @VisibleForTesting
    public static void mergeDesignMatrices(final List<AltSiteRecord> altDesignMatrix, List<AltSiteRecord> altDesignMatrixRevComp){
        if (altDesignMatrix.isEmpty() && altDesignMatrixRevComp.isEmpty()){
            return;
        }

        // Order matters here. Assumes that all elements in the list have the same reference context
        Utils.validateArg(altDesignMatrix.isEmpty() || F1R2FilterConstants.CANONICAL_KMERS.contains(altDesignMatrix.get(0).getReferenceContext()),
                "altDesignMatrix must have the canonical representation");

        final Optional<String> refContext = altDesignMatrix.isEmpty() ? Optional.empty() :
                Optional.of(altDesignMatrix.get(0).getReferenceContext());
        final Optional<String> revCompContext = altDesignMatrixRevComp.isEmpty() ? Optional.empty() :
                Optional.of(altDesignMatrixRevComp.get(0).getReferenceContext());

        // If the matrices aren't empty, their reference context much be the reverse complement of each other
        if (refContext.isPresent() && revCompContext.isPresent()){
            Utils.validateArg(refContext.get().equals(SequenceUtil.reverseComplement(revCompContext.get())),
                    "ref context and its rev comp don't match");
        }

        altDesignMatrix.addAll(altDesignMatrixRevComp.stream().map(AltSiteRecord::getReverseComplementOfRecord).collect(Collectors.toList()));
    }

    public static MetricsFile<?, Integer> readMetricsFile(File file){
        final MetricsFile<?, Integer> metricsFile = new MetricsFile<>();
        final Reader reader = IOUtil.openFileForBufferedReading(file);
        metricsFile.read(reader);
        CloserUtil.close(reader);
        return metricsFile;
    }

    public static List<Histogram<Integer>> sumHistogramsFromFiles(final List<MetricsFile<?,Integer>> metricsFiles, final boolean ref){
        Utils.nonNull(metricsFiles, "files may not be null");
        if (metricsFiles.isEmpty()) {
            return Collections.emptyList();
        }

        final List<Histogram<Integer>> histogramList = metricsFiles.get(0).getAllHistograms();
        if (ref){
            Utils.validate(histogramList.size() == F1R2FilterConstants.NUM_KMERS,
                    "The list of ref histograms need to include all kmers as enforced by CollectF1R2Counts");
            Utils.validate(histogramList.stream().allMatch(h -> F1R2FilterConstants.ALL_KMERS.contains(h.getValueLabel())),
                    "a histogram contains an unsupported, non-kmer header");
        } else {
            Utils.validate(histogramList.size() == F1R2FilterConstants.NUM_KMERS * F1R2FilterConstants.numAltHistogramsPerContext,
                    "The list of alt histograms missing some (kmer, alt allele, f1r2) triple");
        }

        for (int i = 1; i < metricsFiles.size(); i++){
            final List<Histogram<Integer>> ithHistograms = metricsFiles.get(i).getAllHistograms();
            for (final Histogram<Integer> jthHistogram : ithHistograms){
                final String refContext = jthHistogram.getValueLabel();
                final Optional<Histogram<Integer>> hist = histogramList.stream().filter(h -> h.getValueLabel().equals(refContext)).findAny();
                Utils.validate(hist.isPresent(),"Missing histogram header for: " + refContext);

                hist.get().addHistogram(jthHistogram);
            }
        }
        return histogramList;
    }

    @VisibleForTesting
    static Map<String, List<AltSiteRecord>> gatherAltSiteRecords(final List<File> tables){
        final Map<String, List<AltSiteRecord>> result = new LinkedHashMap<>();
        for (final File table : tables) {
            final Pair<String, List<AltSiteRecord>> sampleAndRecords = AltSiteRecord.readAltSiteRecords(table.toPath(), DEFAULT_INITIAL_LIST_SIZE);
            final String sample = sampleAndRecords.getLeft();
            final List<AltSiteRecord> records = sampleAndRecords.getRight();

            if (result.containsKey(sample)) {
                result.get(sample).addAll(records);
            } else {
                result.put(sample, records);
            }
        }

        return result;
    }
}