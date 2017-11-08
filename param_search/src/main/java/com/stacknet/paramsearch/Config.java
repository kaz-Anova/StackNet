package com.stacknet.paramsearch;

import picocli.CommandLine;
import picocli.CommandLine.Option;

//Java –jar stacknet.jar train task=classification sparse=false has_head=true model=model
// pred_file=pred.csv train_file=sample_train.csv test_file= sample_test.csv test_target=true
// params=params.txt verbose=true threads=7 metric=logloss stackdata=false seed=1 folds=5 bins=3
public class Config implements Runnable {
    // match string with boolean.
    @Option(names = {"t", "train"}, description = "Training")
    boolean isTrain = true;

    // support both task= and task space to get value
    @Option(names = {"task"}, paramLabel = "<classification>",
            description = "could be either 'regression' or 'classification'.")
    String task;

    static Config getConfig(String[] args) {
        Config config = new Config();

        CommandLine.run(config, System.err, args);
        return config;
    }

    @Option(names = {"sparse"}, paramLabel = "<input data type>",
            description = "If input files are in sparse format. defaults is false")
    private static boolean is_sparse=false;


    @Override
    public void run() {
        System.out.println(isTrain);
        System.out.println(task);

        if (isTrain) {
            System.out.println(isTrain);
        }
    }
}
