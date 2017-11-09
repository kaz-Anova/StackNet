package com.stacknet.paramsearch;

import com.stacknet.paramsearch.utils.IOUtils;
import com.stacknet.paramsearch.utils.ModelUtils;
import crossvalidation.splits.kfold;
import matrix.fsmatrix;

/*
  Store fields needed for run one experiment using csv file to get best params.
 */
class Experiment {
    private final fsmatrix train;
    private final fsmatrix test;
    private final double[] label;
    private final double[] testLabel;

    public Experiment(fsmatrix train, fsmatrix test, double[] label, double[] testLabel) {
        this.train = train;
        this.test = test;
        this.label = label;
        this.testLabel = testLabel;
    }

    public static Experiment getExperimentFromCSV(
            io.input in, String trainFile, String testFile) {
        fsmatrix train = IOUtils.getMatrixFromFile(
                trainFile, in);
        double label []= in.GetTarget(); // we retrieve the label. If we had not set target_columns this would have been null
        fsmatrix test = IOUtils.getMatrixFromFile(
                testFile, in);
        double[] testLabel =in.GetTarget();
        return new Experiment(train,test, label, testLabel);
    }

    // TODO change estimator interface so that we can train any model using same interface.
    public double run(String params) {

        double mean_logloss=0.0; // the metric;
        //specify number of folds
        int folds=5;
        //set a kfolder object
        int kfolder [][][]= kfold.getindices(label.length, folds);

        // begin cross validation
        System.out.println(" begin cross validation");
        for (int f=0; f <folds; f++){

            int train_indices[]=kfolder[f][0]; // train indices
            int test_indices[]=kfolder[f][1]; // test indices
            // create train an cv data based on array's indices
            fsmatrix X_train= train.makerowsubset(train_indices);
            fsmatrix X_cv= train.makerowsubset(test_indices);
            //also slice the target
            double [] y_train=manipulate.select.rowselect.RowSelect(label, train_indices);
            double [] y_cv=manipulate.select.rowselect.RowSelect(label, test_indices);
            System.out.printf(" cv fold: %d/%d  \n", f+1,folds);
            double logloss = ModelUtils.trainRandomForest(
                    params, y_train, X_train, X_cv, y_cv);
            mean_logloss+= logloss;
        }
        mean_logloss/=folds; // average logloss
        System.out.printf(" Final logloss-----> %.4f <-----\n",mean_logloss);
        System.out.println(" Beginning test modelling");

        return ModelUtils.trainRandomForest(params, label, train, test, testLabel);

    }

}
