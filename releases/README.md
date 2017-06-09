# Versions

## V0.2 version 09/06/2017

1.	Added `regression` which can be defined with a new `task` parameter in the command line. Otherwise the user needs to set `classification`. Task is now a mndatory parameter
2.	Added (compiled) Xgboost and created an `XgboostClassifier` and `XgboostRegressor` based on StackNet's api
3.	Fixed baug with lowercase `true` being ignored from parameters
4.	Fixed bug that was not ignoring zeros loaded from sparse data
5.	Added an `equalsizebinner` in `preprocess.binning`
6.	Added a `bins` command in the `train` argument that that allows classifiers to be used in regression problems. The target variable gets binned using `equalsizebinner` and then used in classification.
7.	Added an `rsquared` in metrics
 
## V0.1 version 01/04/2017

1.	Made significant **changes to the ml.tree** module. Previously it was built assuming a dense format. Whereas now it is built having a sparse format at its base. Significant changes have been made to enable hashing to be more efficient (but with a little bit more memory). Dense format might have suffered a bit speed-wise, but it will be improved in the future. 
2.	Improved multithreading in ml.tree
3.	Included a `indices_name` in the command line to print a .csv file `for each fold`  with the corresponding train(0) and valiation(1) indices one stack under the other .The format it `row_index,[0 if training else 1]`. First it prints the train indices and then the validation indices in exactly the same order as they appear when modelling inside StackNet. 
4.	Fixed a bug that was printing the same prediction file twice int he last level of the training process.
5.	Fixed a bug when there was not a target variable (or an extra column)  in the test file
6.	Fixed a bug with reading data in Sparse format
7.	Added standardscaler in code, BUT it is not available in StackNet yet (only in source code)  

