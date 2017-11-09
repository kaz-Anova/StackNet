package com.stacknet.paramsearch;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TrainerTest {
    private static final Logger log = LoggerFactory.getLogger(TrainerTest.class);

    @Test
    public void testTraining() {
        io.input in = TestUtils.getIOWithTarget();
        Experiment e = Experiment.getExperimentFromCSV(
                in,
                TestUtils.getResourceFileAbsolutePath("manual_index/trainb.csv"),
                TestUtils.getResourceFileAbsolutePath("manual_index/testb.csv")
        );
        e.run("bootsrap:false estimators:100 threads:3 offset:0.00001 max_depth:6 max_features:0.4 min_leaf:2.0 min_split:5.0 Objective:ENTROPY row_subsample:0.95 seed:1 verbose:false");

    }
}
