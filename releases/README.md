# Versions

## V0.4 version 31/10/2017

1.	Added [original libfm](http://www.libfm.org/), [libffm classification](https://github.com/guestwalk/libffm) and [vowpal wabbit](https://github.com/JohnLangford/vowpal_wabbit) wrappers. 5 algorithms in total.
2.	Added `input_index` command. This allows StackNet to be run with user-provided indices as a separate file.
3.	Added `include_target` command. It appends the target variable in the beginning of the file when `output_name` is used.
4.	Added the ability to make comments in the `params` files using *#*. Anything on the right of this symbol is regarded as comment. 
5.	Fixed an assertion error in `SklearnknnClassifier`.   
6.	Fixed bug in scoring for (StackNet's implementation of) libfm.   
7.	Fixed other minor errors.


## V0.3 version 06/08/2017

1.	Added Python subprocesses. The user needs to install python himself/herself and have python available on PATH  
2.	Added 22 new algorthms in total (Regerssors and Classifiers)
3.	Added various algorithms based on Sklearn compatible with version 0.18.2. Specifically : SklearnAdaBoostClassifier, SklearnAdaBoostRegressor, SklearnDecisionTreeClassifier, SklearnDecisionTreeRegressor, SklearnExtraTreesClassifier, SklearnExtraTreesRegressor, SklearnknnClassifier, SklearnknnRegressor, SklearnMLPClassifier, SklearnMLPRegressor, SklearnRandomForestClassifier, SklearnRandomForestRegressor, SklearnSGDClassifier, SklearnSGDRegressor, SklearnsvmClassifier, SklearnsvmRegressor
4.	Added support for keras' algorithms through python, compatible with version 2.0.6. It was tested with tehano 0.9.0, but it should work with tf too. The user is responsible for installing keras and for optimizing its backend (and make sure it is available through python). Specifically added KerasnnRegressor or KerasnnClassifier.
5.	Added support for user-defined python scripts. The user can name them as PythonGenericRegressor[index] or PythonGenericClassifier[index] and put them inside lib/python (see PythonGenericClassifier0 example to understand the right structure). He/She could then call it within the parameters as `PythonGenericClassifier index:0`.
6.	Added [Fast_rgf](https://github.com/baidu/fast_rgf) and sepcifically FRGFClassifier and FRGFRegressor
7.	Fixed a back for failling to cast StackNetClassifier as StackNetRegressor when `task=predict`.
8.	Added the display of average metric for all models at the end of each level (eg. average logloss of all folds for each model) 

## V0.2 version 25/06/2017

1.	Added `bagging` as ab extra hyper parameter in each model . It could be specified as `bags:3` 
2.	Added (compiled) lightGBM and created an `LightgbmClassifier` and `LightgbmRegressor` based on StackNet's api
3.	Added [H2O-3](https://github.com/h2oai/h2o-3)'s Algorithms 
4.	Specifically added H2OGbmRegressor, H2ODeepLearningRegressor ,H2ODrfRegressor, H2OGlmRegressor
5.	and H2OGbmClassifier, H2ODeepLearningClassifier ,H2ODrfClassifier, H2OGlmClassifier, H2ONaiveBayesClassifier.
6.	Fixed a bug in `bins` that was causing one less bin to be created
7.	Added BaggingClassifier and BaggingRegressor

## V0.1 version 09/06/2017

1.	Added `regression` which can be defined with a new `task` parameter in the command line. Otherwise the user needs to set `classification`. Task is now a mndatory parameter
2.	Added (compiled) Xgboost and created an `XgboostClassifier` and `XgboostRegressor` based on StackNet's api
3.	Fixed baug with lowercase `true` being ignored from parameters
4.	Fixed bug that was not ignoring zeros loaded from sparse data
5.	Added an `equalsizebinner` in `preprocess.binning`
6.	Added a `bins` command in the `train` argument that that allows classifiers to be used in regression problems. The target variable gets binned using `equalsizebinner` and then used in classification.
7.	Added an `rsquared` in metrics
 
## V0.0 version 01/04/2017

1.	Made significant **changes to the ml.tree** module. Previously it was built assuming a dense format. Whereas now it is built having a sparse format at its base. Significant changes have been made to enable hashing to be more efficient (but with a little bit more memory). Dense format might have suffered a bit speed-wise, but it will be improved in the future. 
2.	Improved multithreading in ml.tree
3.	Included a `indices_name` in the command line to print a .csv file `for each fold`  with the corresponding train(0) and valiation(1) indices one stack under the other .The format it `row_index,[0 if training else 1]`. First it prints the train indices and then the validation indices in exactly the same order as they appear when modelling inside StackNet. 
4.	Fixed a bug that was printing the same prediction file twice int he last level of the training process.
5.	Fixed a bug when there was not a target variable (or an extra column)  in the test file
6.	Fixed a bug with reading data in Sparse format
7.	Added standardscaler in code, BUT it is not available in StackNet yet (only in source code)  

