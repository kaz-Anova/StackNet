package com.stacknet.paramsearch.utils;
import matrix.fsmatrix;
import ml.Tree.RandomForestClassifier;

public class ModelUtils {

    // TODO change estimator interface so that we can train any model using same interface.
    public static double trainRandomForest(
            String modelParams, double [] y_train, fsmatrix X_train, fsmatrix X_cv,double [] y_cv) {
        // the modelling object
        RandomForestClassifier model = new RandomForestClassifier();
        model.set_params(modelParams); //put the parameters based on the string object on the top
        //Set target
        model.target=y_train;

        //fit model
        model.fit(X_train);
        //make predictions in probabilities
        double preds[][]=model.predict_proba(X_cv);
        double logloss=computelogloss(y_cv ,preds); // compute logloss for the current fold	based on the method at the bottom
        System.out.printf("training size: %d test size: %d logloss-----> ( %.4f) \n", X_train.GetRowDimension(),X_cv.GetRowDimension(),logloss);
        return logloss;
    }

    /**
     * @param target : the target variable
     * @param probas : rows/columns probability predictions
     * @return the logloss value
     */
    public static double computelogloss(double target[], double probas[][]){

        double logloss=0.0;

        double epsilon= 1e-15;

        for (int i=0; i < target.length; i++){
            logloss-=Math.log(Math.min(Math.max(probas[i][(int) target[i]],epsilon ),1-epsilon));
        }
        return logloss/(double)((target.length)) ;

    }
}
