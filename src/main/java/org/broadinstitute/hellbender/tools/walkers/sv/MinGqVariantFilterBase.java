package org.broadinstitute.hellbender.tools.walkers.sv;

import htsjdk.variant.variantcontext.Genotype;
import htsjdk.variant.variantcontext.VariantContext;
import htsjdk.variant.vcf.VCFConstants;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.engine.FeatureContext;
import org.broadinstitute.hellbender.engine.ReadsContext;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.engine.VariantWalker;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.utils.samples.PedigreeValidationType;
import org.broadinstitute.hellbender.utils.samples.SampleDBBuilder;
import org.broadinstitute.hellbender.utils.samples.Trio;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.math3.util.FastMath;

import static org.apache.commons.math3.util.FastMath.*;


/**
 * Extract matrix of properties for each variant.
 * Also extract, numVariants x numTrios x 3 tensors of allele count and genotype quality.
 * These data will be used to train a variant filter based on min GQ" (and stratified by other variant properties) that
 * maximizes the admission of variants with Mendelian inheritance pattern while omitting non-Mendelian variants.
 *
 * Derived class must implement abstract method trainFilter()
 */
public abstract class MinGqVariantFilterBase extends VariantWalker {
    @Argument(fullName=StandardArgumentDefinitions.PEDIGREE_FILE_LONG_NAME, shortName=StandardArgumentDefinitions.PEDIGREE_FILE_SHORT_NAME,
            doc="Pedigree file")
    public File pedigreeFile = null;

    @Argument(fullName="keep-homvar", shortName="K", doc="Keep homvar variants even if their GQ is less than min-GQ", optional = true)
    public boolean keepHomvar = false;

    // numVariants x numProperties matrix of variant properties
    protected Map<String, double[]> variantPropertiesMap = null;
    // numVariants x numTrios x 3 tensor of allele counts:
    protected ArrayList<int[][]> alleleCountsTensor = new ArrayList<>();
    // numVariants x numTrios x 3 tensor of genotype qualities:
    protected ArrayList<int[][]> genotypeQualitiesTensor = new ArrayList<>();

    private int numVariants;
    private int numTrios;
    private int numProperties;

    static private final String SVLEN_KEY = "SVLEN";
    static private final String EVIDENCE_KEY = "EVIDENCE";
    static private final String AF_PROPERTY_NAME = "AF";

    // properties used to gather main matrix / tensors during apply()
    private Set<Trio> pedTrios = null;
    private final ArrayList<Double> alleleFrequencies = new ArrayList<>();
    private final ArrayList<String> svTypes = new ArrayList<>();
    private final ArrayList<Integer> svLens = new ArrayList<>();
    private final ArrayList<Set<String>> variantFilters = new ArrayList<>();
    private final ArrayList<Set<String>> variantEvidence = new ArrayList<>();
    // saved initial values
    private List<String> allEvidenceTypes = null;
    private List<String> allFilterTypes = null;
    private List<String> allSvTypes = null;
    private List<String> propertyNames = null;
    private Map<String, Double> propertyBaseline = null;
    private Map<String, Double> propertyScale = null;

    // properties used to estimate pseudo-derivatives of min GQ filter quality
    static final int DOWN_IDX = 0;
    static final int MIDDLE_IDX = 1;
    static final int UP_IDX = 2;
    private int[] minGqForDerivs = null;
    private int[] numPassedForDerivs = null;
    private int[] numMendelianForDerivs = null;
    private float[] d1NumPassed = null;
    private float[] d2NumPassed = null;
    private float[] d1NumMendelian = null;
    private float[] d2NumMendelian = null;
    private float[] d1f1 = null;
    private float[] d2f1 = null;

    protected final int getNumVariants() {
        return numVariants;
    }

    protected final int getNumTrios() {
        return numTrios;
    }

    protected final int getNumProperties() {
        return numProperties;
    }

    protected final List<String> getPropertyNames() {
        return propertyNames;
    }

    /**
     * Entry-point function to initialize the samples database from input data
     */
    private void getPedTrios() {
        final SampleDBBuilder sampleDBBuilder = new SampleDBBuilder(PedigreeValidationType.STRICT);
        sampleDBBuilder.addSamplesFromPedigreeFiles(Collections.singletonList(pedigreeFile));
        pedTrios = sampleDBBuilder.getFinalSampleDB().getTrios();
        if(pedTrios.isEmpty()) {
            throw new UserException.BadInput("The pedigree file must contain trios: " + pedigreeFile);
        }
    }

    @Override
    public void onTraversalStart() {
        getPedTrios();
    }

    private static boolean mapContainsTrio(final Map<String, Integer> map, final Trio trio) {
        return map.containsKey(trio.getPaternalID()) && map.containsKey(trio.getMaternalID())
                && map.containsKey(trio.getChildID());
    }

    private static int[] getMappedTrioProperties(final Map<String, Integer> map, final Trio trio) {
        return new int[] {map.get(trio.getPaternalID()), map.get(trio.getMaternalID()), map.get(trio.getChildID())};
    }


    /**
     * Accumulate properties for variant matrix, and allele counts, genotype quality for trio tensors
     */
    @Override
    public void apply(VariantContext vc, ReadsContext readsContext, ReferenceContext ref, FeatureContext featureContext) {
        // get per-sample allele counts as a map indexed by sample ID
        Map<String, Integer> sampleAlleleCounts = vc.getGenotypes().stream().collect(
                Collectors.toMap(
                        Genotype::getSampleName,
                        g -> g.getAlleles().stream().mapToInt(a -> a.isReference() ? 0 : 1).sum()
                )
        );
        // get the numTrios x 3 matrix of trio allele counts for this variant, keeping only trios where all samples
        // are present in this VariantContext
        int[][] trioAlleleCounts = pedTrios.stream()
                .filter(trio -> mapContainsTrio(sampleAlleleCounts, trio))
                .map(trio -> getMappedTrioProperties(sampleAlleleCounts, trio))
                .collect(Collectors.toList())
                .toArray(new int[0][0]);
        alleleCountsTensor.add(trioAlleleCounts);

        // get per-sample genotype qualities as a map indexed by sample ID
        Map<String, Integer> sampleGenotypeQualities = vc.getGenotypes().stream().collect(
                Collectors.toMap(Genotype::getSampleName, Genotype::getGQ)
        );
        // get the numTrios x 3 matrix of trio genotype qualities for this variant, keeping only trios where all samples
        // are present in this VariantContext
        int[][] trioGenotypeQualities = pedTrios.stream()
                .filter(trio -> mapContainsTrio(sampleGenotypeQualities, trio))
                .map(trio -> getMappedTrioProperties(sampleGenotypeQualities, trio))
                .collect(Collectors.toList()).toArray(new int[0][0]);
        genotypeQualitiesTensor.add(trioGenotypeQualities);

        double alleleFrequency = vc.getAttributeAsDouble(VCFConstants.ALLELE_FREQUENCY_KEY, -1.0);
        if(alleleFrequency <= 0) {
            // VCF not annotated with allele frequency, guess it from allele counts
            final int numAlleles = vc.getGenotypes().stream().mapToInt(Genotype::getPloidy).sum();
            alleleFrequency = sampleAlleleCounts.values().stream().mapToInt(i->i).sum() / (double) numAlleles;
        }
        alleleFrequencies.add(alleleFrequency);

        final String svType = vc.getAttributeAsString(VCFConstants.SVTYPE, null);
        if(svType == null) {
            throw new GATKException("Missing " + VCFConstants.SVTYPE + " for variant " + vc.getID());
        }
        svTypes.add(svType);

        int svLen = vc.getAttributeAsInt(SVLEN_KEY, Integer.MIN_VALUE);
        if(svLen == Integer.MIN_VALUE) {
            throw new GATKException("Missing " + SVLEN_KEY + " for variant " + vc.getID());
        }
        svLens.add(svLen);

        Set<String> vcFilters = vc.getFilters();
        variantFilters.add(vcFilters);

        Set<String> vcEvidence = Arrays.stream(vc.getAttributeAsString(EVIDENCE_KEY, "NO_EVIDENCE")
                .replaceAll("[\\[\\]]", "").split(",")).collect(Collectors.toSet());
        if(vcEvidence.isEmpty()) {
            throw new GATKException("Missing " + EVIDENCE_KEY + " for variant " + vc.getID());
        }
        variantEvidence.add(vcEvidence);
    }

    private double getBaselineOrdered(final double[] orderedValues) {
        // get baseline as median of values
        return orderedValues.length == 0 ?
                0 :
                orderedValues.length % 2 == 1 ?
                        orderedValues[orderedValues.length / 2] :
                        (orderedValues[orderedValues.length / 2 - 1] + orderedValues[orderedValues.length / 2]) / 2.0;
    }

    private double getScaleOrdered(final double[] orderedValues, final double baseline) {
        // get scale as root-mean-square difference from baseline, over central half of data (to exclude outliers)
        switch(orderedValues.length) {
            case 0:
            case 1:
                return 1.0;
            default:
                final int start = orderedValues.length / 4;
                final int stop = 3 * orderedValues.length / 4;
                double scale = 0.0;
                for(int idx = start; idx < stop; ++idx) {
                    scale += (orderedValues[idx] - baseline) * (orderedValues[idx] - baseline);
                }
                return FastMath.max(FastMath.sqrt(scale / (1 + stop - start)), 1.0e-6);
        }
    }

    private static double[] zScore(final double[] values, final double baseline, final double scale) {
        return Arrays.stream(values).map(x -> (x - baseline) / scale).toArray();
    }

    private static double[] zScore(final int[] values, final double baseline, final double scale) {
        return Arrays.stream(values).mapToDouble(x -> (x - baseline) / scale).toArray();
    }

    private static double[] zScore(final boolean[] values, final double baseline, final double scale) {
        return IntStream.range(0, values.length).mapToDouble(i -> ((values[i] ? 1 : 0) - baseline) / scale).toArray();
    }

    private double[] zScoreProperty(final String propertyName, final double[] values) {
        if(propertyBaseline == null) {
            propertyBaseline = new HashMap<>();
        }
        if(propertyScale == null) {
            propertyScale = new HashMap<>();
        }
        if(!propertyBaseline.containsKey(propertyName)) {
            final double[] orderedValues = Arrays.stream(values).sorted().toArray();
            propertyBaseline.put(propertyName, getBaselineOrdered(orderedValues));
            propertyScale.put(propertyName,
                    getScaleOrdered(orderedValues, propertyBaseline.get(propertyName)));
        }
        final double baseline = propertyBaseline.get(propertyName);
        final double scale = propertyScale.get(propertyName);
        return zScore(values, baseline, scale);
    }

    private double[] zScoreProperty(final String propertyName, final int[] values) {
        if(propertyBaseline == null) {
            propertyBaseline = new HashMap<>();
        }
        if(propertyScale == null) {
            propertyScale = new HashMap<>();
        }
        if(!propertyBaseline.containsKey(propertyName)) {
            final double[] orderedValues = Arrays.stream(values).sorted().mapToDouble(i -> i).toArray();
            propertyBaseline.put(propertyName, getBaselineOrdered(orderedValues));
            propertyScale.put(propertyName,
                    getScaleOrdered(orderedValues, propertyBaseline.get(propertyName)));
        }
        final double baseline = propertyBaseline.get(propertyName);
        final double scale = propertyScale.get(propertyName);
        return zScore(values, baseline, scale);
    }

    private double[] zScoreProperty(final String propertyName, final boolean[] values) {
        if(propertyBaseline == null) {
            propertyBaseline = new HashMap<>();
        }
        if(propertyScale == null) {
            propertyScale = new HashMap<>();
        }
        if(!propertyBaseline.containsKey(propertyName)) {
            final long numTrue = IntStream.range(0, values.length).filter(i -> values[i]).count();
            final long numFalse = values.length - numTrue;
            final double baseline = numTrue / (double)values.length;
            final double scale = numTrue == 0 || numFalse == 0 ?
                    1.0 : FastMath.sqrt(numTrue * numFalse / (values.length * (double)values.length));
            propertyBaseline.put(propertyName, baseline);
            propertyScale.put(propertyName, scale);
        }
        final double baseline = propertyBaseline.get(propertyName);
        final double scale = propertyScale.get(propertyName);
        return zScore(values, baseline, scale);
    }

    private List<String> assignAllLabels(final List<String> labelsList, List<String> allLabels) {
        return allLabels == null ?
               labelsList.stream().sorted().distinct().collect(Collectors.toList()) :
               allLabels;
    }

    private List<String> assignAllSetLabels(final List<Set<String>> labelsList, List<String> allLabels) {
        return allLabels == null ?
               labelsList.stream().flatMap(Set::stream).sorted().distinct().collect(Collectors.toList()) :
               allLabels;
    }

    private Map<String, double[]> labelsToLabelStatus(final List<String> labels, List<String> allLabels) {
        return labelsListsToLabelStatus(
                labels.stream().map(Collections::singleton).collect(Collectors.toList()),
                allLabels
        );
    }

    private Map<String, double[]> labelsListsToLabelStatus(final List<Set<String>> labelsList, List<String> allLabels) {
        final Map<String, boolean[]> labelStatus = allLabels.stream()
                .collect(Collectors.toMap(
                        label -> label, label -> new boolean[labelsList.size()]
                ));
        int variantIdx = 0;
        for (final Set<String> variantLabels : labelsList) {
            final int idx = variantIdx; // need final or "effectively final" variable for lambda expression
            variantLabels.forEach(label -> labelStatus.get(label)[idx] = true);
            ++variantIdx;
        }
        return labelStatus.entrySet().stream().collect(Collectors.toMap(
                Map.Entry::getKey,
                e -> zScoreProperty(e.getKey(), e.getValue())
        ));
    }

    private void collectVariantPropertiesMap() {
        numVariants = alleleFrequencies.size();
        if(numVariants == 0) {
            throw new GATKException("No variants contained in vcf: " + drivingVariantFile);
        }
        numTrios = alleleCountsTensor.get(0).length;
        if(numTrios == 0) {
            throw new UserException.BadInput("There are no trios from the pedigree file that are fully represented in the vcf");
        }

        allEvidenceTypes = assignAllSetLabels(variantEvidence, allEvidenceTypes);
        allFilterTypes = assignAllSetLabels(variantFilters, allFilterTypes);
        allSvTypes = assignAllLabels(svTypes, allSvTypes);
        variantPropertiesMap = Stream.of(
                labelsListsToLabelStatus(variantEvidence, allEvidenceTypes),
                labelsListsToLabelStatus(variantFilters, allFilterTypes),
                labelsToLabelStatus(svTypes, allSvTypes),
                Collections.singletonMap(
                        AF_PROPERTY_NAME, zScoreProperty(AF_PROPERTY_NAME, alleleFrequencies.stream().mapToDouble(x -> x).toArray())
                ),
                Collections.singletonMap(
                        SVLEN_KEY, zScoreProperty(SVLEN_KEY, svLens.stream().mapToInt(x -> x).toArray())
                )
        ).flatMap(e -> e.entrySet().stream()).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

        List<String> suppliedPropertyNames = propertyNames == null ? null : new ArrayList<>(propertyNames);
        propertyNames = variantPropertiesMap.keySet().stream().sorted().collect(Collectors.toList());
        if(suppliedPropertyNames != null && !suppliedPropertyNames.equals(propertyNames)) {
            throw new GATKException("Extracted properties not compatible with supplied propertyNames");
        }
        numProperties = propertyNames.size();
    }



    void printDebugInfo() {
        System.out.println("numVariants: " + numVariants);
        System.out.println("numTrios: " + numTrios);
        System.out.println("numProperties: " + numProperties);
        System.out.println("index\tpropertyName\tpropertyBaseline\tpropertyScale");
        int idx = 0;
        for(final String propertyName : propertyNames) {
            System.out.println(idx + "\t" + propertyName + "\t" + propertyBaseline.get(propertyName) + "\t" + propertyScale.get(propertyName));
            ++idx;
        }
        idx = 0;
        for(final String filterType : allFilterTypes) {
            System.out.println(idx + "\t" + filterType);
            ++idx;
        }

        idx = 0;
        for(final String evidenceType : allEvidenceTypes) {
            System.out.println(idx + "\t" + evidenceType);
            ++idx;
        }
        idx = 0;
        for(final String svType : allSvTypes) {
            System.out.println(idx + "\t" + svType);
            ++idx;
        }
    }


    private boolean isMendelian(final int fatherAc, final int motherAc, final int childAc) {
        // child allele counts should not exhibit de-novo mutations nor be missing inherited homvar
        int minAc = fatherAc / 2 + motherAc / 2;
        int maxAc = (fatherAc + 1) / 2 + (motherAc + 1) / 2;
        return (minAc <= childAc) && (childAc <= maxAc);
    }

    private int filterAlleleCount(final int ac, final int gq, final int minGq) {
        return gq >= minGq || (ac == 2 && keepHomvar) ? ac : 0;
    }

    protected float getMinGqQuality(final int[] minGq) {
        long numChecks = 0;
        long numMendelian = 0;
        long numPassed = 0;
        for(int variantIdx = 0; variantIdx < numVariants; ++variantIdx) {
            final int variantMinGq = minGq[variantIdx];
            for(int trioIdx = 0; trioIdx < numTrios; ++trioIdx) {
                final int[] trioAc = alleleCountsTensor.get(variantIdx)[trioIdx];
                final int numAllelesTrio = trioAc[0] + trioAc[1] + trioAc[2];
                if(numAllelesTrio == 0) {
                    continue;
                }
                numChecks += numAllelesTrio;
                final int[] trioGq = genotypeQualitiesTensor.get(variantIdx)[trioIdx];
                final int fatherAc = filterAlleleCount(trioAc[0], trioGq[0], variantMinGq);
                final int motherAc = filterAlleleCount(trioAc[1], trioGq[1], variantMinGq);
                final int childAc = filterAlleleCount(trioAc[2], trioGq[2], variantMinGq);
                final int numPassedTrio = fatherAc + motherAc + childAc;
                if(numPassedTrio > 0) {
                    numPassed += numPassedTrio;
                    if(isMendelian(fatherAc, motherAc, childAc)) {
                        numMendelian += numPassedTrio;
                    }
                }
            }
        }
        final float recall = numMendelian / (float) numChecks;
        final float precision = numMendelian / (float) numPassed;
        // return max f1-score
        return numMendelian == 0 ? (float)0 : 2 * recall * precision / (recall + precision);
    }

    private int checkTrioMinGqQualities(final int[][] variantAc, final int[][] variantGq, final int bracketIdx,
                                         final int[] minGq, final int[] numPassed, final int[] numMendelian) {
        final int variantMinGq = minGq[bracketIdx];
        numPassed[bracketIdx] = 0;
        numMendelian[bracketIdx] = 0;
        int numChecks = 0;
        for(int trioIdx = 0; trioIdx < numTrios; ++trioIdx) {
            final int[] trioAc = variantAc[trioIdx];
            final int numAllelesTrio = trioAc[0] + trioAc[1] + trioAc[2];
            if(numAllelesTrio == 0) {
                continue;
            }
            numChecks += numAllelesTrio;
            final int[] trioGq = variantGq[trioIdx];
            final int fatherAc = filterAlleleCount(trioAc[0], trioGq[0], variantMinGq);
            final int motherAc = filterAlleleCount(trioAc[1], trioGq[1], variantMinGq);
            final int childAc = filterAlleleCount(trioAc[2], trioGq[2], variantMinGq);
            final int numPassedTrio = fatherAc + motherAc + childAc;
            if(numPassedTrio > 0) {
                numPassed[bracketIdx] += numPassedTrio;
                if(isMendelian(fatherAc, motherAc, childAc)) {
                    numMendelian[bracketIdx] += numPassedTrio;
                }
            }
        }
        return numChecks;
    }

    private void setFiniteDifferenceGq(final int[][] variantGq, final int[][] variantAc, final int[] minGqForDerivs) {
        final int[] gqArr = Arrays.stream(variantGq).flatMapToInt(Arrays::stream).toArray();
        final int[] acArr = Arrays.stream(variantAc).flatMapToInt(Arrays::stream).toArray();
        final int[] candidateGq = IntStream.range(0, gqArr.length)
                .filter(keepHomvar ? i -> acArr[i] == 1 : i -> acArr[i] > 0)
                .map(i -> gqArr[i])
                .sorted()
                .distinct()
                .toArray();
        minGqForDerivs[DOWN_IDX] = Integer.MIN_VALUE;
        minGqForDerivs[UP_IDX] = Integer.MAX_VALUE;
        final int midGq = minGqForDerivs[MIDDLE_IDX];
        for(final int checkGq : candidateGq) {
            if(checkGq < midGq && minGqForDerivs[DOWN_IDX] < checkGq) {
                minGqForDerivs[DOWN_IDX] = checkGq;
            } else if(checkGq > midGq && minGqForDerivs[UP_IDX] > checkGq) {
                minGqForDerivs[UP_IDX] = checkGq;
            }
        }
    }

    private float sqr(final float x) { return x * x; }

    private void setDerivsFromDifferences(final float xTrue, final int[] x, final int[] y,
                                          final int variantIdx, final float[] d1, final float[] d2) {
        // model as quadratic: y = a * x^2 + b * x + c
        final float denom = (float)((x[2] - x[1]) * (x[2] - x[0]) * (x[1] - x[0]));
        final float a = ((x[2] - x[1]) * (y[1] - y[0]) - (x[1] - x[0]) * (y[2] - y[1])) / denom;
        final float b = ((sqr(x[2]) - sqr(x[1])) * (y[1] - y[0]) - (sqr(x[1]) - sqr(x[0])) * (y[2] - y[1])) / denom;
        final float d2_x = 2 * a;
        d1[variantIdx] = d2_x * xTrue + b;
        if(d2 != null) {
            d2[variantIdx] = d2_x;
        }
    }

    protected float getMinGqQuality(final float[] minGq, final int numDerivatives) {
        if(minGqForDerivs == null) {
            minGqForDerivs = new int[3];
            numPassedForDerivs = new int[3];
            numMendelianForDerivs = new int[3];
        }
        if(numDerivatives > 0) {
            if(d1f1 == null) {
                d1f1 = new float[numVariants];
                d1NumPassed = new float[numVariants];
                d1NumMendelian = new float[numVariants];
            }
            if(numDerivatives > 1) {
                if(numDerivatives > 2) {
                    throw new GATKException("Maximum value for numDerivatives is 2. Supplied value is " + numDerivatives);
                }
                if(d2f1 == null) {
                    d2f1 = new float[numVariants];
                    d2NumPassed = new float[numVariants];
                    d2NumMendelian = new float[numVariants];
                }
            }
        }

        long numChecks = 0;
        long numPassed = 0;
        long numMendelian = 0;
        for(int variantIdx = 0; variantIdx < numVariants; ++variantIdx) {
            final int[][] variantAc = alleleCountsTensor.get(variantIdx);
            final int[][] variantGq = genotypeQualitiesTensor.get(variantIdx);

            final float variantMinGq = minGq[variantIdx];
            minGqForDerivs[MIDDLE_IDX] = (int)ceil(variantMinGq);
            numChecks += checkTrioMinGqQualities(variantAc, variantGq, MIDDLE_IDX,
                                                 minGqForDerivs, numPassedForDerivs, numMendelianForDerivs);
            numPassed += numPassedForDerivs[MIDDLE_IDX];
            numMendelian += numMendelianForDerivs[MIDDLE_IDX];

            if(numDerivatives > 0) {
                setFiniteDifferenceGq(variantGq, variantAc, minGqForDerivs);
                if (minGqForDerivs[DOWN_IDX] == Integer.MIN_VALUE) {
                    // there's no possible change at lower min GQ. Set same values at one lower min GQ
                    minGqForDerivs[DOWN_IDX] = minGqForDerivs[MIDDLE_IDX] - 1;
                    numPassedForDerivs[DOWN_IDX] = numPassedForDerivs[MIDDLE_IDX];
                    numMendelianForDerivs[DOWN_IDX] = numMendelianForDerivs[MIDDLE_IDX];
                } else {
                    checkTrioMinGqQualities(variantAc, variantGq, DOWN_IDX,
                                            minGqForDerivs, numPassedForDerivs, numMendelianForDerivs);
                }
                if (minGqForDerivs[UP_IDX] == Integer.MAX_VALUE) {
                    // there's no possible change at higher min GQ. Set same values at one lower min GQ
                    minGqForDerivs[UP_IDX] = minGqForDerivs[MIDDLE_IDX] + 1;
                    numPassedForDerivs[UP_IDX] = numPassedForDerivs[MIDDLE_IDX];
                    numMendelianForDerivs[UP_IDX] = numMendelianForDerivs[MIDDLE_IDX];
                } else {
                    checkTrioMinGqQualities(variantAc, variantGq, UP_IDX,
                                            minGqForDerivs, numPassedForDerivs, numMendelianForDerivs);
                }
                setDerivsFromDifferences(variantMinGq, minGqForDerivs, numPassedForDerivs, variantIdx,
                                         d1NumPassed, numDerivatives > 1 ? d2NumPassed : null);
                setDerivsFromDifferences(variantMinGq, minGqForDerivs, numMendelianForDerivs, variantIdx,
                                         d1NumMendelian, numDerivatives > 1 ? d2NumMendelian : null);
            }
        }

        // calculate f1 score:
        //     f1 = 2.0 / (1.0 / recall + 1.0 / precision)
        //     recall = numMendelian / numChecks
        //     precision = numMendelian / numPassed
        //     -> f1 = 2.0 * numMendelian / (numChecks + numPassed)
        final float denom = (float)(numChecks + numPassed);
        final float f1 = 2 * numMendelian / denom;
        if(numDerivatives > 0) {
            for(int variantIdx = 0; variantIdx < numDerivatives; ++variantIdx) {
                d1f1[variantIdx] = (2 * d1NumMendelian[variantIdx] - f1 * d1NumPassed[variantIdx]) / denom;
                if(numDerivatives == 2) {
                    d2f1[variantIdx] = (2 * d2NumMendelian[variantIdx] - f1 * d2NumPassed[variantIdx]
                                        - 2 * d1NumPassed[variantIdx] * d1f1[variantIdx] / denom) / denom;
                }
            }
        }
        return f1;
    }


    protected abstract void trainFilter();

    @Override
    public Object onTraversalSuccess() {
        collectVariantPropertiesMap();

        printDebugInfo();

        trainFilter();
        return null;
    }
}
