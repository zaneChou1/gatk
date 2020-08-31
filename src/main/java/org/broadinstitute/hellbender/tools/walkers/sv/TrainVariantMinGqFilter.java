package org.broadinstitute.hellbender.tools.walkers.sv;

import htsjdk.variant.variantcontext.Genotype;
import htsjdk.variant.variantcontext.VariantContext;
import htsjdk.variant.vcf.VCFConstants;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.barclay.help.DocumentedFeature;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.cmdline.programgroups.StructuralVariantDiscoveryProgramGroup;
import org.broadinstitute.hellbender.engine.FeatureContext;
import org.broadinstitute.hellbender.engine.ReadsContext;
import org.broadinstitute.hellbender.engine.ReferenceContext;
import org.broadinstitute.hellbender.engine.VariantWalker;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.utils.samples.PedigreeValidationType;
import org.broadinstitute.hellbender.utils.samples.SampleDBBuilder;
import org.broadinstitute.hellbender.utils.samples.Trio;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.math3.util.FastMath;


@CommandLineProgramProperties(
        summary = "Extract properties for each variant, along with allele count. Train filter to accept or reject variants based on " +
                  "Genotype Quality, concordance with Mendelian inheritance (from allele counts and pedigree file) as an error function.",
        oneLineSummary = "Train min GQ filter from .vcf and .ped file with trios.",
        programGroup = StructuralVariantDiscoveryProgramGroup.class
)
@DocumentedFeature
public class TrainVariantMinGqFilter extends VariantWalker {
    @Argument(fullName= StandardArgumentDefinitions.PEDIGREE_FILE_LONG_NAME, shortName=StandardArgumentDefinitions.PEDIGREE_FILE_SHORT_NAME,
            doc="Pedigree file", optional=false)
    private File pedigreeFile = null;

    // num_variants x num_properties matrix of variant properties
    private double[][] variant_properties_matrix;
    // num_variants x num_trios x 3 tensor of allele counts:
    private ArrayList<int[][]> allele_counts_tensor = new ArrayList<>();
    // num_variants x num_trios x 3 tensor of genotype qualities:
    private ArrayList<int[][]> genotype_qualities_tensor = new ArrayList<>();

    private int num_variants;
    private int num_trios;
    private int num_properties;

    static private final String SVLEN_KEY = "SVLEN";
    static private final String EVIDENCE_KEY = "EVIDENCE";
    static private final String AF_PROPERTY_NAME = "AF";

    // properties used to gather main matrix / tensors during apply()
    private Set<Trio> trios = null;
    private ArrayList<Double> allele_frequencies = new ArrayList<>();
    private ArrayList<String> sv_types = new ArrayList<>();
    private ArrayList<Integer> sv_lens = new ArrayList<>();
    private ArrayList<Set<String>> variant_filters = new ArrayList<>();
    private ArrayList<Set<String>> variant_evidence = new ArrayList<>();
    // saved initial values
    private List<String> all_evidence_types = null;
    private List<String> all_filter_types = null;
    private List<String> all_sv_types = null;
    private List<String> property_names = null;
    private Map<String, Double> property_baseline = null;
    private Map<String, Double> property_scale = null;

    /**
     * Entry-point function to initialize the samples database from input data
     */
    private Set<Trio> getTrios() {
        final SampleDBBuilder sampleDBBuilder = new SampleDBBuilder(PedigreeValidationType.STRICT);
        sampleDBBuilder.addSamplesFromPedigreeFiles(Collections.singletonList(pedigreeFile));
        return sampleDBBuilder.getFinalSampleDB().getTrios();
    }

    @Override
    public void onTraversalStart() {
        trios = getTrios();
    }

    private static boolean map_contains_trio(final Map<String, Integer> map, final Trio trio) {
        return map.containsKey(trio.getPaternalID()) && map.containsKey(trio.getMaternalID())
                && map.containsKey(trio.getChildID());
    }

    private static int[] get_mapped_trio_properties(final Map<String, Integer> map, final Trio trio) {
        return new int[] {map.get(trio.getPaternalID()), map.get(trio.getMaternalID()), map.get(trio.getChildID())};
    }

    private double get_baseline_ordered(final double[] ordered_values) {
        // get baseline as median of values
        return ordered_values.length == 0 ?
                0 :
                ordered_values.length % 2 == 1 ?
                        ordered_values[ordered_values.length / 2] :
                        (ordered_values[ordered_values.length / 2 - 1] + ordered_values[ordered_values.length / 2]) / 2.0;
    }

    private double get_scale_ordered(final double[] ordered_values, final double baseline) {
        // get scale as root-mean-square difference from baseline, over central half of data (to exclude outliers)
        switch(ordered_values.length) {
            case 0:
            case 1:
                return 1.0;
            default:
                final int start = ordered_values.length / 4;
                final int stop = 3 * ordered_values.length / 4;
                double scale = 0.0;
                for(int idx = start; idx < stop; ++idx) {
                    scale += (ordered_values[idx] - baseline) * (ordered_values[idx] - baseline);
                }
                return FastMath.max(FastMath.sqrt(scale / (1 + stop - start)), 1.0e-6);
        }
    }

    private static double[] z_score(final double[] values, final double baseline, final double scale) {
        return Arrays.stream(values).map(x -> (x - baseline) / scale).toArray();
    }

    private static double[] z_score(final int[] values, final double baseline, final double scale) {
        return Arrays.stream(values).mapToDouble(x -> (x - baseline) / scale).toArray();
    }

    private static double[] z_score(final boolean[] values, final double baseline, final double scale) {
        return IntStream.range(0, values.length).mapToDouble(i -> ((values[i] ? 1 : 0) - baseline) / scale).toArray();
    }

    private double[] z_score_property(final String property_name, final double[] values) {
        if(property_baseline == null) {
            property_baseline = new HashMap<>();
        }
        if(property_scale == null) {
            property_scale = new HashMap<>();
        }
        if(!property_baseline.containsKey(property_name)) {
            final double[] ordered_values = Arrays.stream(values).sorted().toArray();
            property_baseline.put(property_name, get_baseline_ordered(ordered_values));
            property_scale.put(property_name,
                    get_scale_ordered(ordered_values, property_baseline.get(property_name)));
        }
        final double baseline = property_baseline.get(property_name);
        final double scale = property_scale.get(property_name);
        return z_score(values, baseline, scale);
    }

    private double[] z_score_property(final String property_name, final int[] values) {
        if(property_baseline == null) {
            property_baseline = new HashMap<>();
        }
        if(property_scale == null) {
            property_scale = new HashMap<>();
        }
        if(!property_baseline.containsKey(property_name)) {
            final double[] ordered_values = Arrays.stream(values).sorted().mapToDouble(i -> i).toArray();
            property_baseline.put(property_name, get_baseline_ordered(ordered_values));
            property_scale.put(property_name,
                    get_scale_ordered(ordered_values, property_baseline.get(property_name)));
        }
        final double baseline = property_baseline.get(property_name);
        final double scale = property_scale.get(property_name);
        return z_score(values, baseline, scale);
    }

    private double[] z_score_property(final String property_name, final boolean[] values) {
        if(property_baseline == null) {
            property_baseline = new HashMap<>();
        }
        if(property_scale == null) {
            property_scale = new HashMap<>();
        }
        if(!property_baseline.containsKey(property_name)) {
            final long num_true = IntStream.range(0, values.length).filter(i -> values[i]).count();
            final long num_false = values.length - num_true;
            final double baseline = num_true / (double)values.length;
            final double scale = num_true == 0 || num_false == 0 ?
                    1.0 : FastMath.sqrt(num_true * num_false / (values.length * (double)values.length));
            property_baseline.put(property_name, baseline);
            property_scale.put(property_name, scale);
        }
        final double baseline = property_baseline.get(property_name);
        final double scale = property_scale.get(property_name);
        return z_score(values, baseline, scale);
    }

    private List<String> assign_all_labels(final List<String> labels_list, List<String> all_labels) {
        return all_labels == null ?
                labels_list.stream().sorted().distinct().collect(Collectors.toList()) :
                all_labels;
    }

    private List<String> assign_all_set_labels(final List<Set<String>> labels_list, List<String> all_labels) {
        return all_labels == null ?
                labels_list.stream().flatMap(Set::stream).sorted().distinct().collect(Collectors.toList()) :
                all_labels;
    }

    private Map<String, double[]> labels_to_label_status(final List<String> labels, List<String> all_labels) {
        return labels_lists_to_label_status(
                labels.stream().map(Collections::singleton).collect(Collectors.toList()),
                all_labels
        );
    }

    private Map<String, double[]> labels_lists_to_label_status(final List<Set<String>> labels_list, List<String> all_labels) {
        final Map<String, boolean[]> label_status = all_labels.stream()
                .collect(Collectors.toMap(
                        label -> label, label -> new boolean[labels_list.size()]
                ));
        int variant_idx = 0;
        for (final Set<String> variant_labels : labels_list) {
            final int idx = variant_idx; // need final or "effectively final" variable for lambda expression
            variant_labels.forEach(label -> label_status.get(label)[idx] = true);
            ++variant_idx;
        }
        return label_status.entrySet().stream().collect(Collectors.toMap(
                Map.Entry::getKey,
                e -> z_score_property(e.getKey(), e.getValue())
        ));
    }

    private void pack_variant_properties_matrix() {
        num_variants = allele_frequencies.size();
        if(num_variants == 0) {
            throw new GATKException("No variants contained in " + drivingVariantFile);
        }
        num_trios = allele_counts_tensor.get(0).length;

        // create the variant properties matrix
        // 1. Compute and z-score the following properties
        //    present-or-not for variant_evidence, variant_filters, sv_types
        //    ordinal values for allele_frequencies, sv_lens
        // 2. Stream them together, and sort by property name
        // 3. Store property names in List
        // 4. Concatenate z-scored values into a matrix

        all_evidence_types = assign_all_set_labels(variant_evidence, all_evidence_types);
        all_filter_types = assign_all_set_labels(variant_filters, all_filter_types);
        all_sv_types = assign_all_labels(sv_types, all_sv_types);
        final Map<String, double[]> variant_properties_map = Stream.of(
                labels_lists_to_label_status(variant_evidence, all_evidence_types),
                labels_lists_to_label_status(variant_filters, all_filter_types),
                labels_to_label_status(sv_types, all_sv_types),
                Collections.singletonMap(
                        AF_PROPERTY_NAME, z_score_property(AF_PROPERTY_NAME, allele_frequencies.stream().mapToDouble(x -> x).toArray())
                ),
                Collections.singletonMap(
                        SVLEN_KEY, z_score_property(SVLEN_KEY, sv_lens.stream().mapToInt(x -> x).toArray())
                )
        ).flatMap(e -> e.entrySet().stream()).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

        List<String> supplied_property_names = property_names == null ? null : new ArrayList<>(property_names);
        property_names = variant_properties_map.keySet().stream().sorted().collect(Collectors.toList());
        if(supplied_property_names != null && !supplied_property_names.equals(property_names)) {
            throw new GATKException("Extracted properties not compatible with supplied property_names");
        }
        num_properties = property_names.size();

        variant_properties_matrix = new double[num_variants][num_properties];
        for (int col = 0; col < num_properties; ++col) {
            final String property_name = property_names.get(col);
            final double[] property_values = variant_properties_map.get(property_name);
            for(int row = 0; row < num_variants; ++row) {
                variant_properties_matrix[row][col] = property_values[row];
            }
        }

        num_properties = variant_properties_matrix[0].length;
    }

    @Override
    public void apply(VariantContext vc, ReadsContext readsContext, ReferenceContext ref, FeatureContext featureContext) {
        // get per-sample allele counts as a map indexed by sample ID
        Map<String, Integer> sample_allele_counts = vc.getGenotypes().stream().collect(
                Collectors.toMap(
                        Genotype::getSampleName,
                        g -> g.getAlleles().stream().mapToInt(a -> a.isReference() ? 0 : 1).sum()
                )
        );
        // get the num_trios x 3 matrix of trio allele counts for this variant, keeping only trios where all samples
        // are present in this VariantContext
        int[][] trio_allele_counts = trios.stream()
                .filter(trio -> map_contains_trio(sample_allele_counts, trio))
                .map(trio -> get_mapped_trio_properties(sample_allele_counts, trio))
                .collect(Collectors.toList())
                .toArray(new int[0][0]);
        allele_counts_tensor.add(trio_allele_counts);

        // get per-sample genotype qualities as a map indexed by sample ID
        Map<String, Integer> sample_genotype_qualities = vc.getGenotypes().stream().collect(
                Collectors.toMap(Genotype::getSampleName, Genotype::getGQ)
        );
        // get the num_trios x 3 matrix of trio genotype qualities for this variant, keeping only trios where all samples
        // are present in this VariantContext
        int[][] trio_genotype_qualities = trios.stream()
                .filter(trio -> map_contains_trio(sample_genotype_qualities, trio))
                .map(trio -> get_mapped_trio_properties(sample_genotype_qualities, trio))
                .collect(Collectors.toList()).toArray(new int[0][0]);
        genotype_qualities_tensor.add(trio_genotype_qualities);

        double allele_frequency = vc.getAttributeAsDouble(VCFConstants.ALLELE_FREQUENCY_KEY, -1.0);
        if(allele_frequency <= 0) {
            // VCF not annotated with allele frequency, guess it from allele counts
            final int num_alleles = vc.getGenotypes().stream().mapToInt(Genotype::getPloidy).sum();
            allele_frequency = sample_allele_counts.values().stream().mapToInt(i->i).sum() / (double) num_alleles;
        }
        allele_frequencies.add(allele_frequency);

        final String sv_type = vc.getAttributeAsString(VCFConstants.SVTYPE, null);
        if(sv_type == null) {
            throw new GATKException("Missing " + VCFConstants.SVTYPE + " for variant " + vc.getID());
        }
        sv_types.add(sv_type);

        int sv_len = vc.getAttributeAsInt(SVLEN_KEY, Integer.MIN_VALUE);
        if(sv_len == Integer.MIN_VALUE) {
            throw new GATKException("Missing " + SVLEN_KEY + " for variant " + vc.getID());
        }
        sv_lens.add(sv_len);

        Set<String> vc_filters = vc.getFilters();
        variant_filters.add(vc_filters);

        Set<String> vc_evidence = Arrays.stream(vc.getAttributeAsString(EVIDENCE_KEY, "NO_EVIDENCE")
                .replaceAll("[\\[\\]]", "").split(",")).collect(Collectors.toSet());
        if(vc_evidence.isEmpty()) {
            throw new GATKException("Missing " + EVIDENCE_KEY + " for variant " + vc.getID());
        }
        variant_evidence.add(vc_evidence);
    }

    void print_debug_info() {
        System.out.println("num_variants: " + num_variants);
        System.out.println("num_trios: " + num_trios);
        System.out.println("num_properties: " + num_properties);
        System.out.println("index\tproperty_name\tproperty_baseline\tproperty_scale");
        int idx = 0;
        for(final String property_name : property_names) {
            System.out.println(idx + "\t" + property_name + "\t" + property_baseline.get(property_name) + "\t" + property_scale.get(property_name));
            ++idx;
        }
        idx = 0;
        for(final String filter_type : all_filter_types) {
            System.out.println(idx + "\t" + filter_type);
            ++idx;
        }

        idx = 0;
        for(final String evidence_type : all_evidence_types) {
            System.out.println(idx + "\t" + evidence_type);
            ++idx;
        }
        idx = 0;
        for(final String sv_type : all_sv_types) {
            System.out.println(idx + "\t" + sv_type);
            ++idx;
        }

    }

    @Override
    public Object onTraversalSuccess() {
        pack_variant_properties_matrix();

        print_debug_info();
        return null;
    }
}
