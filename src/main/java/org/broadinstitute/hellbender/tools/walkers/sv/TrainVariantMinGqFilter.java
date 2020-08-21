package org.broadinstitute.hellbender.tools.walkers.sv;

import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.barclay.help.DocumentedFeature;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.cmdline.programgroups.StructuralVariantDiscoveryProgramGroup;
import org.broadinstitute.hellbender.engine.VariantWalker;
import org.broadinstitute.hellbender.utils.samples.PedigreeValidationType;
import org.broadinstitute.hellbender.utils.samples.SampleDB;
import org.broadinstitute.hellbender.utils.samples.SampleDBBuilder;

import java.io.File;
import java.util.Collections;

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

    private SampleDB sampleDB = null;
    /**
     * Entry-point function to initialize the samples database from input data
     */
    private SampleDB initializeSampleDB() {
        final SampleDBBuilder sampleDBBuilder = new SampleDBBuilder(PedigreeValidationType.STRICT);
        sampleDBBuilder.addSamplesFromPedigreeFiles(Collections.singletonList(pedigreeFile));
        return sampleDBBuilder.getFinalSampleDB();
    }

    @Override
    public void onTraversalStart() {
        sampleDB = initializeSampleDB();

    }
}
