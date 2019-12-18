package org.broadinstitute.hellbender.tools.sv;

import htsjdk.samtools.SAMSequenceDictionary;
import htsjdk.samtools.util.IOUtil;
import htsjdk.tribble.Tribble;
import htsjdk.tribble.index.Index;
import htsjdk.tribble.index.IndexFactory;
import htsjdk.variant.variantcontext.StructuralVariantType;
import htsjdk.variant.variantcontext.VariantContext;
import htsjdk.variant.vcf.VCFHeader;
import htsjdk.variant.vcf.VCFHeaderLine;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.barclay.argparser.ExperimentalFeature;
import org.broadinstitute.barclay.help.DocumentedFeature;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.cmdline.programgroups.StructuralVariantDiscoveryProgramGroup;
import org.broadinstitute.hellbender.engine.FeatureDataSource;
import org.broadinstitute.hellbender.engine.GATKPath;
import org.broadinstitute.hellbender.engine.GATKTool;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.tools.copynumber.PostprocessGermlineCNVCalls;
import org.broadinstitute.hellbender.utils.IntervalUtils;
import org.broadinstitute.hellbender.utils.codecs.SVCallRecordCodec;
import org.broadinstitute.hellbender.utils.io.IOUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Creates sparsely formatted file of structural variants.
 *
 * <h3>Inputs</h3>
 *
 * <ul>
 *     <li>
 *         Standardized SV VCFs
 *     </li>
 *     <li>
 *         gCNV segments VCFs
 *     </li>
 *     <li>
 *         cnMOPs call tables
 *     </li>
 * </ul>
 *
 * <h3>Output</h3>
 *
 * <ul>
 *     <li>
 *         TSV variants file
 *     </li>
 *     <li>
 *         Small CNV interval list (optional)
 *     </li>
 * </ul>
 *
 * <h3>Usage example</h3>
 *
 * <pre>
 *     gatk MergeSVCalls
 * </pre>
 *
 * @author Mark Walker &lt;markw@broadinstitute.org&gt;
 */

@CommandLineProgramProperties(
        summary = "Creates sparse structural variants file",
        oneLineSummary = "Creates sparse structural variants file",
        programGroup = StructuralVariantDiscoveryProgramGroup.class
)
@ExperimentalFeature
@DocumentedFeature
public final class MergeSVCalls extends GATKTool {
    public static final String MIN_GCNV_QUALITY_LONG_NAME = "min-gcnv-quality";
    public static final String SMALL_CNV_SIZE_LONG_NAME = "small-cnv-size";
    public static final String SMALL_CNV_PADDING_LONG_NAME = "small-cnv-padding";
    public static final String SMALL_CNV_OUTPUT_LONG_NAME = "small-cnv-output";
    public static final String IGNORE_DICTIONARY_LONG_NAME = "ignore-dict";
    public static final String CNMOPS_INPUT_LONG_NAME = "cnmops";
    public static final String COMPRESSION_LEVEL_LONG_NAME = "compression-level";
    public static final String CREATE_INDEX_LONG_NAME = "create-index";

    public static final int DEFAULT_SMALL_CNV_SIZE = 5000;
    public static final int DEFAULT_SMALL_CNV_PADDING = 1000;

    @Argument(
            doc = "Input standardized SV and gCNV segments VCFs",
            fullName = StandardArgumentDefinitions.VARIANT_LONG_NAME,
            shortName = StandardArgumentDefinitions.VARIANT_SHORT_NAME
    )
    private List<String> inputFiles;

    @Argument(
            doc = "cnMOPS input files in tabular format",
            fullName = CNMOPS_INPUT_LONG_NAME
    )
    private List<String> cnmopsFiles;

    @Argument(
            doc = "Output file ending in \"" + SVCallRecordCodec.FORMAT_SUFFIX + "\" or \"" + SVCallRecordCodec.COMPRESSED_FORMAT_SUFFIX + "\"",
            fullName = StandardArgumentDefinitions.OUTPUT_LONG_NAME,
            shortName = StandardArgumentDefinitions.OUTPUT_SHORT_NAME
    )
    private GATKPath outputFile;

    @Argument(fullName=CREATE_INDEX_LONG_NAME,
            doc = "If true, create a index when writing output file", optional=true, common = true)
    public boolean createOutputIndex = true;

    @Argument(
            doc = "Skip VCF sequence dictionary check",
            fullName = IGNORE_DICTIONARY_LONG_NAME,
            optional = true
    )
    private Boolean skipVcfDictionaryCheck = false;

    @Argument(
            doc = "Min gCNV quality (QS)",
            fullName = MIN_GCNV_QUALITY_LONG_NAME,
            minValue = 0,
            maxValue = Integer.MAX_VALUE,
            optional = true
    )
    private int minGCNVQuality = 60;

    @Argument(
            doc = "Compression level for gzipped output",
            fullName = COMPRESSION_LEVEL_LONG_NAME
    )
    private int compressionLevel = 6;

    private List<SVCallRecord> records;
    private SAMSequenceDictionary dictionary;
    private static final SVCallRecordCodec CALL_RECORD_CODEC = new SVCallRecordCodec();

    @Override
    public void onTraversalStart() {
        dictionary = getBestAvailableSequenceDictionary();
        if (dictionary == null) {
            throw new UserException("Reference sequence dictionary required");
        }
        records = new ArrayList<>();
    }

    @Override
    public Object onTraversalSuccess() {
        records.sort(IntervalUtils.getDictionaryOrderComparator(dictionary));
        writeVariants();
        if (createOutputIndex) {
            writeIndex();
        }
        return null;
    }

    @Override
    public void traverse() {
        for (final String path : cnmopsFiles) {
            processCNMOPSFile(path);
        }
        for (int i = 0; i < inputFiles.size(); i++) {
            processVariantFile(inputFiles.get(i), i);
        }
    }

    private void processCNMOPSFile(final String path) {
        try {
            final BufferedReader reader = new BufferedReader(IOUtils.makeReaderMaybeGzipped(Paths.get(path)));
            final String headerString = reader.readLine();
            if (headerString == null) {
                logger.warn("Skipping empty cnMOPS file: " + path);
                return;
            }
            logger.info("Parsing cnMOPS file: " + path);
            if (!headerString.startsWith("#")) {
                throw new UserException.BadInput("Expected first line to be a header starting with \"#\" but found: \"" + headerString + "\"");
            }
            reader.lines().map(this::cnmopsRecordParser).forEach(r -> {
                records.add(r);
                progressMeter.update(r.getStartAsInterval());
            });
        } catch (final IOException e) {
            throw new UserException.CouldNotReadInputFile("Encountered exception while reading cnMOPS file: " + path, e);
        }
    }

    private SVCallRecord cnmopsRecordParser(final String line) {
        final String[] tokens = line.split("\t");
        final String contig = tokens[0];
        final int start = Integer.valueOf(tokens[1]);
        final int end = Integer.valueOf(tokens[2]);
        final Set<String> samples = Collections.singleton(tokens[4]);
        final StructuralVariantType type = StructuralVariantType.valueOf(tokens[5]);
        if (!type.equals(StructuralVariantType.DEL) && !type.equals(StructuralVariantType.DUP)) {
            throw new UserException.BadInput("Unexpected cnMOPS record type: " + type.name());
        }
        final int length = end - start;
        final boolean isDel = type.equals(StructuralVariantType.DEL);
        final boolean startStrand = isDel;
        final boolean endStrand = !isDel;
        final List<String> algorithms = Collections.singletonList(SVCluster.DEPTH_ALGORITHM);
        return new SVCallRecord(contig, start, startStrand, contig, end, endStrand, type, length, algorithms, samples);
    }

    private void processVariantFile(final String file, final int index) {
        final FeatureDataSource<VariantContext> source = getFeatureDataSource(file, "file_" + index);
        final VCFHeader header = getHeaderFromFeatureSource(source);
        final VCFHeaderLine headerSource = header.getMetaDataLine("source");
        final Stream<VariantContext> inputVariants = StreamSupport.stream(source.spliterator(), false);
        final Stream<SVCallRecord> inputRecords;
        if (headerSource != null && headerSource.getValue().equals(PostprocessGermlineCNVCalls.class.getSimpleName())) {
            inputRecords = inputVariants
                    .map(v -> SVCallRecord.createDepthOnlyFromGCNV(v, minGCNVQuality))
                    .filter(r -> r != null);
        } else {
            inputRecords = inputVariants.map(SVCallRecord::create);
        }
        inputRecords.forEachOrdered(r -> {
            records.add(r);
            progressMeter.update(r.getStartAsInterval());
        });
    }

    private VCFHeader getHeaderFromFeatureSource(final FeatureDataSource<VariantContext> source) {
        final Object header = source.getHeader();
        if ( ! (header instanceof VCFHeader) ) {
            throw new GATKException("Header for " + source.getName() + " is not in VCF header format");
        }
        return (VCFHeader)header;
    }

    private FeatureDataSource<VariantContext> getFeatureDataSource(final String vcf, final String name) {
        final FeatureDataSource<VariantContext> featureDataSource = new FeatureDataSource<>(
                vcf, name, 100000, VariantContext.class, cloudPrefetchBuffer, cloudIndexPrefetchBuffer);
        if (!skipVcfDictionaryCheck) {
            featureDataSource.getSequenceDictionary().assertSameDictionary(dictionary);
        }
        featureDataSource.setIntervalsForTraversal(getTraversalIntervals());
        return featureDataSource;
    }

    private void writeVariants() {
        try (final PrintStream writer =  IOUtils.makePrintStreamMaybeBlockGzipped(outputFile.toPath().toFile(), compressionLevel)) {
            for (final SVCallRecord record : records) {
                writer.println(CALL_RECORD_CODEC.encode(record));
            }
        } catch(final IOException e) {
            throw new GATKException("Error writing output file", e);
        }
    }

    private void writeIndex() {
        final Path outPath = outputFile.toPath();
        try {
            final Index index;
            final Path indexPath;
            if (IOUtil.hasBlockCompressedExtension(outPath)) {
                index = IndexFactory.createIndex(outPath, CALL_RECORD_CODEC, IndexFactory.IndexType.TABIX, dictionary);
                indexPath = Tribble.tabixIndexPath(outPath);
            } else {
                // Optimize indices for other kinds of files for seek time / querying
                index = IndexFactory.createDynamicIndex(outPath, CALL_RECORD_CODEC, IndexFactory.IndexBalanceApproach.FOR_SEEK_TIME);
                indexPath = Tribble.indexPath(outPath);
            }
            index.write(indexPath);
        } catch (final IOException e) {
            throw new UserException.CouldNotIndexFile(outputFile.toPath(), e);
        }
    }
}
