package stacknetrun;

import org.junit.Assert;
import org.junit.Test;
import picocli.CommandLine;
import stacknetrun.Config;

public class ConfigTest {

    @Test
    public void testCommand() {
        String[] args = {"","train","task","tt"};
        Config config = Config.getConfig(args);

        Assert.assertEquals(config.is_train, false);
        Assert.assertEquals(config.task, "tt");
    }
}
