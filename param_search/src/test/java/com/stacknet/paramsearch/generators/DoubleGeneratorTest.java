package com.stacknet.paramsearch.generators;

import com.google.common.base.Joiner;
import com.stacknet.paramsearch.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DoubleGeneratorTest {
    private static final Logger log = LoggerFactory.getLogger(DoubleGeneratorTest.class);

    @Test
    public void getGridDoubleGenerator() {
        DoubleGenerator dg = DoubleGenerator.getGridDoubleGenerator(0.0, 1.0);
        double[] expected = {0.0,0.2,0.4,0.6,0.8};
        Assert.assertArrayEquals(expected,
                TestUtils.toArray(dg.expend(5)), 0.001);
    }

    @Test
    public void getUniformDoubleGenerator() {
        DoubleGenerator dg = DoubleGenerator.getUniformDoubleGenerator(0.0, 1.0);
        String joined = Joiner.on(",").join(dg.expend(5));

        System.out.println(joined);
    }
}
