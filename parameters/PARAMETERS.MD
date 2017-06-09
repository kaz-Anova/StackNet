# Parameters

This Section explains which parameters to tune for each algorithm. Almost all algorithms have in common the following:

Parameter | Explanation
--- | ---
seed | Int value to replicate randomized processes
verbose | If True it prints stuff regarding the progress of an algorithm
threads | Int value to apply parallelism. Not always applicable, but can facilitate speed’s performance
usescale | If True it use maximum absolute scaling. It is useful for linear algorithms
copy | If True, it makes a hard copy of the data. 

## Classifiers

Classifier Models are described first.

### DecisionTreeClassifier

Parameter | Explanation
--- | ---
max_depth | Maximum depth of the tree (double). **This is important**.
Objective | The objective to optimise in split. It may be “ENTROPY “, “GINI” or “AUC”. ENTROPY (default) almost always performs best. **This is important**.
row_subsample | Proportion of observations to consider (double). **This is important**.
max_features | Proportion of columns (features) to consider in each level (double). **This is important**.
cut_off_subsample | Proportion of best cut offs to consider. This controls how *Extremely Randomized* the tree will be (double).
feature_subselection | Proportion of columns (features) to consider for the whole tree (double).
min_leaf | Minimum weighted sum to keep after splitting node (double).
min_split | Minimum weighted sum to split a node (double).
rounding |Digits of rounding to prevent overfitting. It could help in certain situations(double).
max_tree_size | Maximum number of nodes allowed (int)
offset | Adds a constant when calculating the objective in a split. It prevents overfitting (double).
 
The rest of the parameters may be unstable and better left as is. 

### RandomForestClassifier

Parameter | Explanation
--- | ---
estimators | Number of trees to build. In most situations after 100 it does not improve dramatically more (int) .
 max_depth | maximum depth of the tree (double). **This is important**.
Objective | The objective to optimise in split. It may be “ENTROPY “, “GINI” or “AUC”. ENTROPY (default) almost always performs best. **This is important**.
row_subsample | Proportion of observations to consider (double). **This is important**.
max_features | Proportion of columns (features) to consider in each level (double). **This is important**.
cut_off_subsample | Proportion of best cut offs to consider. This controls how *Extremely Randomized* the tree will be (double).
feature_subselection | Proportion of columns (features) to consider for the whole tree (double).
min_leaf | Minimum weighted sum to keep after splitting node (double).
min_split | Minimum weighted sum to split a node (double).
rounding |Digits of rounding to prevent overfitting. It could help in certain situations(double).
max_tree_size | Maximum number of nodes allowed (int)
offset | Adds a constant when calculating the objective in a split. It prevents overfitting (double).

The rest of the parameters may be left as is. 

### AdaboostRandomForestClassifier

Parameter | Explanation
--- | ---
estimators | Number of Random Forests to build. In most situations after 100 it does not improve dramatically more (int) .
trees | Number of trees in each Forest. The default is 1 which basically connotes a adatreeclassifier (int).
weight_thresold | Affects the weight (importance) of each new estimator via setting this initial threshold.  This may be regarded as a shrinkage parameter. Needs to be between 0 and 1 (double). **This is important**.
 max_depth | Maximum depth of the tree (double). **This is important**.
Objective | The objective to optimise in split. It may be “ENTROPY “, “GINI” or “AUC”. ENTROPY (default) almost always performs best. **This is important**.
row_subsample | Proportion of observations to consider (double). **This is important**.
max_features | Proportion of columns (features) to consider in each level (double). **This is important**.
cut_off_subsample | Proportion of best cut offs to consider. This controls how *Extremely Randomized* the tree will be (double).
feature_subselection | Proportion of columns (features) to consider for the whole tree (double).
min_leaf | Minimum weighted sum to keep after splitting node (double).
min_split | Minimum weighted sum to split a node (double).
rounding |Digits of rounding to prevent overfitting. It could help in certain situations(double).
max_tree_size | Maximum number of nodes allowed (int)
offset | Adds a constant when calculating the objective in a split. It prevents overfitting (double).

The rest of the parameters may be left as is. 


### GradientBoostingForestClassifier

Parameter | Explanation
--- | ---
estimators | Number of Random Forests to build. In most situations after 100 it does not improve dramatically more (int) .
trees | Number of trees in each Forest. The default is 1 which basically connotes a adatreeclassifier (int).
shrinkage | Penalty applied to each estimator . Smaller values prevent overfitting.  Needs to be between 0 and 1 (double).  There is also a fairly linear negative correlation between estimators and shrinkage. **This is important**.
 max_depth | Maximum depth of the tree (double). **This is important**.
Objective | The objective to optimise inside the split. It may be “RMSE“ or “MAE”. Bear in mind the underlying estimators are regressors. 
row_subsample | Proportion of observations to consider (double). **This is important**.
max_features | Proportion of columns (features) to consider in each level (double). **This is important**.
cut_off_subsample | Proportions of best cut offs to consider. This controls how *Extremely Randomized* the tree will be. Very low value means only a few cut-offs are explored (double).
feature_subselection | Proportions of columns (features) to consider for the whole tree (double).
min_leaf | Minimum weighted sum to keep after splitting node (double).
min_split | Minimum weighted sum to split a node (double).
rounding |Digits of rounding to prevent overfitting. It could help in certain situations (double).
max_tree_size | Maximum number of nodes allowed (int) .
offset | Adds a constant when calculating the objective in a split. It prevents overfitting (double).

The rest of the parameters may be left as is. 

### LogisticRegression

Parameter | Explanation
--- | ---
C  | Regularization value, the more, the stronger the regularization(double). **This is important**.
l1C  | L1 Regularization C value for FTRL Type (double).
Type | Can be one of “Liblinear”, “Routine”, “SGD”, “FTRL”. Default is Liblinear. SGD and FTRL use adagrad. Routine is based on Matrix multiplications and the Newton-Raphson method.  
RegularizationType | Can be either "L2" or “L1”. Default is “L2”. “L1” is only supported via Liblnear and FTRL. **This is important**.
learn_rate | For SGD and FTRL (double).
UseConstant | If true it uses an intercept.
maxim_Iteration | Maximum number of iterations (int) .
shuffle | True to train on random rows. 

### LSVC

Linear Support Vector Machine for Classification.

Parameter | Explanation
--- | ---
C  | Regularization value, the more, the stronger the regularization(double). **This is important**.
l1C  | L1 Regularization C value for FTRL Type (double).
Type | Can be one of “Liblinear”, “SGD”, “FTRL”. Default is Liblinear. SGD and FTRL use adagrad. 
RegularizationType | Can be either "L2" or “L1”. Default is “L2”. “L1” is only supported via Liblnear and FTRL. **This is important**.
learn_rate | For SGD and FTRL (double).
UseConstant | If true it uses an intercept.
maxim_Iteration | Maximum number of iterations (int) .
shuffle | True to train on random rows. 

### LibFmClassifier

Based on Steffen Rendle’s [libfm] (http://www.libfm.org/)

Parameter | Explanation
--- | ---
C | Regularization value, the more, the stronger the regularization (double). **This is important**.
C2 | Regularization value for the latent features (double). **This is important**.
Lfeatures | Number of latent features to use. Defaults to 4 (int). **This is important**.
init_values  | Initialise values of the latent features with random values between [0,init_values) (double). **This is important**.
learn_rate | For SGD (double). **This is important**.
maxim_Iteration | Maximum number of iterations (int) . **This is important**.
Type | Only “SGD”. 
UseConstant | If true it uses an intercept.
shuffle | True to train on random rows. 

### Softmaxnnclassifier

This is a neural network with 2 hidden layers. It is heavily based on the equivalent one in the [kaggler](https://pypi.python.org/pypi/Kaggler) python package.

Parameter | Explanation
--- | ---
C | Regularization value, the more, the stronger the regularization (double). **This is important**.
h1| Number of the 1st level hidden units (int). **This is important**.
h2| Number of the 2nd level hidden units (int). **This is important**.
init_values  | Initialise values of hidden units with random values between [0,init_values) (double). **This is important**.
smooth | Value to divide gradients and aid convergence (double). **This is important**.
connection_nonlinearity | Can be one of “Relu”,”Linear”,”Sigmoid”,”Tanh”. Commonly  Relu performs best. **This is important**.
learn_rate | For SGD (double). **This is important**.
maxim_Iteration | Maximum number of iterations (int) . **This is important**.
Type | Only “SGD”. 
UseConstant | If true it uses an intercept.
shuffle | True to train on random rows. 

### NaiveBayesClassifier

Parameter | Explanation
--- | ---
Shrinkage | Can be seen as a form of a penalty to avoid really big product’s failures. 

## Regressors

### DecisionTreeRegressor

Parameter | Explanation
--- | ---
max_depth | Maximum depth of the tree (double). **This is important**.
Objective | The objective to optimise in split. It may be “RMSE “ or “MAE”. 
row_subsample | Proportion of observations to consider (double). **This is important**.
max_features | Proportion of columns (features) to consider in each level (double). **This is important**.
cut_off_subsample | Proportion of best cut offs to consider. This controls how *Extremely Randomized* the tree will be (double).
feature_subselection | Proportion of columns (features) to consider for the whole tree (double).
min_leaf | Minimum weighted sum to keep after splitting node (double).
min_split | Minimum weighted sum to split a node (double).
rounding |Digits of rounding to prevent overfitting. It could help in certain situations(double).
max_tree_size | Maximum number of nodes allowed (int)
offset | Adds a constant when calculating the objective in a split. It prevents overfitting (double).
 
The rest of the parameters may be unstable and better left as is. 

### RandomForestRegressor

Parameter | Explanation
--- | ---
estimators | Number of trees to build. In most situations after 100 it does not improve dramatically more (int) .
 max_depth | Maximum depth of the tree (double). **This is important**.
Objective | The objective to optimise in split. It may be “RMSE “ or “MAE”. 
row_subsample | Proportion of observations to consider (double). **This is important**.
max_features | Proportion of columns (features) to consider in each level (double). **This is important**.
cut_off_subsample | Proportion of best cut offs to consider. This controls how *Extremely Randomized* the tree will be (double).
feature_subselection | Proportion of columns (features) to consider for the whole tree (double).
min_leaf | Minimum weighted sum to keep after splitting node (double).
min_split | Minimum weighted sum to split a node (double).
rounding |Digits of rounding to prevent overfitting. It could help in certain situations(double).
max_tree_size | Maximum number of nodes allowed (int)
offset | Adds a constant when calculating the objective in a split. It prevents overfitting (double).

The rest of the parameters may be left as is. 

### AdaboostRandomForestRegressor

Parameter | Explanation
--- | ---
estimators | Number of Random Forests to build. In most situations after 100 it does not improve dramatically more (int) .
trees | Number of trees in each Forest. The default is 1 which basically connotes a adatreeregressor (int).
weight_thresold | Affects the weight (importance) of each new estimator via setting this initial threshold.  This may be regarded as a shrinkage parameter. Needs to be positive (double). **This is important**.
max_depth | Maximum depth of the tree (double). **This is important**.
Objective | The objective to optimise in split. It may be “RMSE “ or “MAE”.
row_subsample | Proportion of observations to consider (double). **This is important**.
max_features | Proportion of columns (features) to consider in each level (double). **This is important**.
cut_off_subsample | Proportion of best cut offs to consider. This controls how *Extremely Randomized* the tree will be (double).
feature_subselection | Proportion of columns (features) to consider for the whole tree (double).
min_leaf | Minimum weighted sum to keep after splitting node (double).
min_split | Minimum weighted sum to split a node (double).
rounding |Digits of rounding to prevent overfitting. It could help in certain situations(double).
max_tree_size | Maximum number of nodes allowed (int)
offset | Adds a constant when calculating the objective in a split. It prevents overfitting (double).

The rest of the parameters may be left as is. 

### GradientBoostingForestRegressor

Parameter | Explanation
--- | ---
estimators | Number of Random Forests to build. In most situations after 100 it does not improve dramatically more (int) .
trees |Number of trees in each Forest. The default is 1 which basically connotes a adatreeclassifier (int).
shrinkage | Penalty applied to each estimator . Smaller values prevent overfitting.  Needs to be between 0 and 1 (double).  There is also a fairly linear negative correlation between estimators and shrinkage. **This is important**.
max_depth | Maximum depth of the tree (double). **This is important**.
Objective | The objective to optimise inside the split. It may be “RMSE“ or “MAE”. 
row_subsample | Proportion of observations to consider (double). **This is important**.
max_features | Proportion of columns (features) to consider in each level (double). **This is important**.
cut_off_subsample | Proportions of best cut offs to consider. This controls how *Extremely Randomized* the tree will be. Very low value means only a few cut-offs are explored (double).
feature_subselection | Proportions of columns (features) to consider for the whole tree (double).
min_leaf | Minimum weighted sum to keep after splitting node (double).
min_split | Minimum weighted sum to split a node (double).
rounding |Digits of rounding to prevent overfitting. It could help in certain situations (double).
max_tree_size | Maximum number of nodes allowed (int) .
offset | Adds a constant when calculating the objective in a split. It prevents overfitting (double).

The rest of the parameters may be left as is. 


### LinearRegression

Parameter | Explanation
--- | ---
C  | Regularization value, the more, the stronger the regularization(double). A value here basically triggers a *Ridge regression*.  **This is important**.
l1C  | L1 Regularization C value for FTRL Type (double).
Type | Can be one of “Routine”, “SGD” or “FTRL”. SGD and FTRL use adagrad. Routine is the Ordinary Least Squares method which is solved with matrix multiplications.  
Objective | Can be one of “RMSE”, “MAE” or ”QUANTILE”.
tau | Tau value for QUANTILE (double).
learn_rate | For SGD and FTRL (double).
UseConstant | If true it uses an intercept.
maxim_Iteration | Maximum number of iterations (int) .
shuffle | True to train on random rows. 

### LSVR

Linear Support Vector Machine for Regression.

Parameter | Explanation
--- | ---
C  | Regularization value, the more, the stronger the regularization(double). **This is important**.
l1C  | L1 Regularization C value for FTRL Type (double).
Type | Can be one of “Liblinear”, “SGD”, “FTRL”. Default is Liblinear. SGD and FTRL use adagrad. 
Objective | Can be either “L1” or “L2” for normal hinge loss and quadratic loss respectively. 
learn_rate | For SGD and FTRL (double).
smooth | value to aid convergence .
UseConstant | If true it uses an intercept.
maxim_Iteration | Maximum number of iterations (int) .
shuffle | True to train on random rows. 

### LibFmRegressor

Based on Steffen Rendle’s [libfm] (http://www.libfm.org/)

Parameter | Explanation
--- | ---
C | Regularization value, the more, the stronger the regularization (double). **This is important**.
C2 | Regularization value for the latent features (double). **This is important**.
Lfeatures | Number of latent features to use. Defaults to 4 (int). **This is important**.
init_values | Initialise values of the latent features with random values between [0,init_values) (double). **This is important**.
learn_rate | For SGD (double). **This is important**.
maxim_Iteration | Maximum number of iterations (int) . **This is important**.
Objective | Can be one of “RMSE”, “MAE” or ”QUANTILE”.
tau | Tau value for QUANTILE (double).
Type | Only “SGD”. 
UseConstant | If true it uses an intercept.
shuffle | True to train on random rows. 

### Multinnregressor

This is a neural network with 2 hidden layers. It is heavily based on the equivalent one in the [kaggler](https://pypi.python.org/pypi/Kaggler) python package.

Parameter | Explanation
--- | ---
C | Regularization value, the more, the stronger the regularization (double). **This is important**.
h1| Number of the 1st level hidden units (int). **This is important**.
h2| Number of the 2nd level hidden units (int). **This is important**.
init_values  | Initialise values of hidden units with random values between [0,init_values) (double). **This is important**.
smooth | Value to divide gradients and aid convergence (double). **This is important**.
connection_nonlinearity | Can be one of “Relu”,”Linear”,”Sigmoid”,”Tanh”. Commonly Relu performs best. **This is important**.
learn_rate | For SGD (double). **This is important**.
maxim_Iteration | Maximum number of iterations (int). **This is important**.
Objective | Can be one of “RMSE”, “MAE” or ”QUANTILE”.
tau | Tau value for QUANTILE (double).
Type | Only “SGD”. 
UseConstant | If true it uses an intercept.
shuffle | True to train on random rows.

### XgboostRegressor(**NEW**)

The original parameters can be found [here](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md) 

Parameter | Explanation
--- | ---
num_round | Number of estimators to build (int) .
max_leaves |Maximum leaves in a tree (int).
eta | Penalty applied to each estimator. Needs to be between 0 and 1 (double). **This is important**.
max_depth | Maximum depth of the tree (int). **This is important**.
Objective | Can be one of ['reg:linear','count:poisson','reg:gamma' ,'rank:pairwise','reg:tweedie']. Note that rank:pairwise is not a regressor but its output was more convenient for a regerssion method.  
subsample | Proportion of observations to consider (double). **This is important**.
colsample_bylevel | Proportion of columns (features) to consider in each level (double).
colsample_bytree |  Proportion of columns (features) to consider in each Tree (double) **This is important**.
max_delta_step | controls optimization step (double).
gamma | controls minimum change requirements in loss to allow for a split (double).
booster | 'gbtree' or 'gblinear'.
alpha |controls overfitting (double). 
lambda |controls overfitting (double). 

### XgboostClassifier(**NEW**)

Same parameters with regression apart from Objective. The user does not need to set anything - the algorithm decideds automatically based on the number of classes in the target variable.

Parameter | Explanation
--- | ---
scale_pos_weight | used for imbalanced classes(double)



