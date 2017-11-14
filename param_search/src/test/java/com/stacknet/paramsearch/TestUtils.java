package com.stacknet.paramsearch;

import com.stacknet.paramsearch.utils.IOUtils;
import matrix.fsmatrix;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class TestUtils {
    private static final Logger log = LoggerFactory.getLogger(TestUtils.class);
    // return InputStream for path under resources folder
    public static InputStream getResourceFileStream(String path) {
        return TestUtils.class.getClassLoader().getResourceAsStream(path);
    }

    public static io.input getIOWithTarget() {
        return IOUtils.getIOWithTarget(",", true, new int[] {0}, 1, 5000);
    }

    public static io.input getIOWithID() {
        return IOUtils.getIOWithID(",", true, 0, 1, 5000);

    }

    public static fsmatrix getMatrixFromResourceFile(
            String file, io.input in){
        String path = TestUtils.getResourceFileAbsolutePath(file);
        log.info(String.format("load %s", file));

        fsmatrix matrix = IOUtils.getMatrixFromFile(
                path, in);
        log.info("Loaded train data with " + matrix.GetRowDimension() + " and columns " + matrix.GetColumnDimension() );
        return matrix;
    }

    public static String getResourceFileAbsolutePath(String path) {
        URL resource = TestUtils.class.getClassLoader().getResource(path);
        try {
            URI u = resource.toURI();
            Path p = Paths.get(u);
            return p.toFile().getAbsolutePath();
        } catch (URISyntaxException e) {
            e.printStackTrace();
            return "";
        }
    }

    public static double[] toArray(List<Double> list) {
        return list.stream().mapToDouble(Number::doubleValue).toArray();
    }

    @Test
    public void testPath() {
        String file = "manual_index/test.csv";
        String p = getResourceFileAbsolutePath(file);
        Assert.assertTrue(p.endsWith(file));
    }
}
