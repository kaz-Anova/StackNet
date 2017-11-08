package com.stacknet.paramsearch;

import org.junit.Assert;
import org.junit.Test;

public class ConfigTest {

    @Test
    public void testCommand() {
        String[] args = {"","train","task","tt"};
        Config config = Config.getConfig(args);

        Assert.assertEquals(config.isTrain, false);
        Assert.assertEquals(config.task, "tt");
    }
}
