package com.stacknet.paramsearch.utils;

import matrix.fsmatrix;

public class IOUtils {
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
}
