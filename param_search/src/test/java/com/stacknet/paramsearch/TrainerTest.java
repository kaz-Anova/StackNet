package com.stacknet.paramsearch;

import com.stacknet.paramsearch.utils.ModelUtils;
import crossvalidation.splits.kfold;
import matrix.fsmatrix;
import ml.Tree.RandomForestClassifier;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;

public class TrainerTest {
    private static final Logger log = LoggerFactory.getLogger(TrainerTest.class);

    @Test
    public void testTraining() {
        io.input in = TestUtils.getIOWithTarget();

        fsmatrix X= TestUtils.getMatrixFromResourceFile(
                "manual_index/trainb.csv", in);
        double label []= in.GetTarget(); // we retrieve the label. If we had not set target_columns this would have been null
        fsmatrix xTest = TestUtils.getMatrixFromResourceFile(
                "manual_index/testb.csv", in);
        double[] testLabel =in.GetTarget();

        double mean_logloss=0.0; // the metric;
        //specify number of folds
        int folds=5;
        //set a kfolder object
        int kfolder [][][]= kfold.getindices(label.length, folds);

        // begin cross validation
        System.out.println(" begin cross validation");
        //model parameters for an RandomForestClassifier models
        String model_params="bootsrap:false estimators:100 threads:3 offset:0.00001 max_depth:6 max_features:0.4 min_leaf:2.0 min_split:5.0 Objective:ENTROPY row_subsample:0.95 seed:1 verbose:false";
        for (int f=0; f <folds; f++){

            int train_indices[]=kfolder[f][0]; // train indices
            int test_indices[]=kfolder[f][1]; // test indices
            // create train an cv data based on array's indices
            fsmatrix X_train= X.makerowsubset(train_indices);
            fsmatrix X_cv= X.makerowsubset(test_indices);
            //also slice the target
            double [] y_train=manipulate.select.rowselect.RowSelect(label, train_indices);
            double [] y_cv=manipulate.select.rowselect.RowSelect(label, test_indices);
            System.out.printf(" cv fold: %d/%d  \n", f+1,folds);
            double logloss = ModelUtils.trainRandomForest(
                    model_params, y_train, X_train, X_cv, y_cv);
            mean_logloss+= logloss;
        }
        mean_logloss/=folds; // average logloss
        System.out.printf(" Final logloss-----> %.4f <-----\n",mean_logloss);
        System.out.println(" Beginning test modelling");

        ModelUtils.trainRandomForest(model_params, label, X, xTest, testLabel);
    }
}
