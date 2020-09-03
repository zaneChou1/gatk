package org.broadinstitute.hellbender.tools.walkers.sv;

import ml.dmlc.xgboost4j.java.XGBoostError;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.barclay.help.DocumentedFeature;
import org.broadinstitute.hellbender.cmdline.programgroups.StructuralVariantDiscoveryProgramGroup;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;
import org.broadinstitute.hellbender.exceptions.GATKException;

import java.util.Arrays;
import java.util.stream.IntStream;

import static java.util.stream.IntStream.*;

@CommandLineProgramProperties(
        summary = "Extract matrix of properties for each variant. Also extract, num_variants x num_trios x 3 tensors of" +
                "allele count and genotype quality. These data will be used to train a variant filter based on min GQ" +
                "(and stratified by other variant properties) that maximizes the admission of variants with Mendelian" +
                "inheritance pattern while omitting non-Mendelian variants." +
                "Derived class must implement abstract method train_filter()",
        oneLineSummary = "Extract data for training min GQ variant filter from .vcf and .ped file with trios.",
        programGroup = StructuralVariantDiscoveryProgramGroup.class
)
@DocumentedFeature
public class MinGqVariantFilter extends MinGqVariantFilterBase {


    private DMatrix getDMatrix() {
        return getDMatrix(null);
    }

    private DMatrix getDMatrix(int[] rowIndices) {
        if(rowIndices == null) {
            rowIndices = IntStream.range(0, getNumVariants()).toArray();
        }
        final int numRows = rowIndices.length;
        final float[] rowMajor = new float[numRows * getNumProperties()];
        int flatIndex = 0;
        for(final int rowIndex : rowIndices) {
            for(final String propertyName : getPropertyNames()) {
                rowMajor[flatIndex] = (float)variantPropertiesMap.get(propertyName)[rowIndex];
                ++flatIndex;
            }
        }

        try {
            return new DMatrix(rowMajor, numRows, getNumProperties());
        }
        catch(XGBoostError e) {
            throw new GATKException(e.getMessage());
        }
    }

    @Override
    protected void trainFilter() {

    }
}
