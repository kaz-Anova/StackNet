package com.stacknet.paramsearch.utils;
import matrix.fsmatrix;
import ml.Tree.RandomForestClassifier;
import ml.estimator;

public class ModelUtils {

    //
    public static io.input getIOWithTarget(
            String delimiter,
            boolean hasHeader,
            int[] targetColumns,
            int startCol,
            int endCol) {
        io.input in = getIO(delimiter, hasHeader, startCol, endCol);
        in.targets_columns= targetColumns; // the first column is the target
        return in;
    }

    // this is data without target but with id
    public static io.input getIOWithID(
            String delimiter,
            boolean hasHeader,
            int id,
            int startCol,
            int endCol) {
        io.input in = getIO(delimiter, hasHeader, startCol, endCol);
        in.idint= id; // the first column is the id and we set it as int, this has to be int value, not array
        return in;
    }

    private static io.input getIO(
            String delimiter,
            boolean hasHeader,
            int startCol,
            int endCol) {
        io.input in = new io.input(); //open a reader-type of class
        in.delimeter= delimiter; // set delimiter
        in.HasHeader=hasHeader; // it does not have headers
        in.start=startCol; //we load the predictors from (1,onward). so we set for the main data everything apart from the target (column 0)
        in.end=endCol;
        return in;
    }

    public static fsmatrix getMatrixFromFile(
            String file,
            io.input in){
        return in.Readfmatrix(file); // we read the data as fixed-size matrix
    }

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
