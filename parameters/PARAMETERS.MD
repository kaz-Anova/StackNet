# Parameters

This Section explains which parameters to tune for each algorithm. Almost all algorithms have in common the following:

Parameter | Explanation
--- | ---
seed | Int value to replicate randomized processes
bags(**new**) | Int value to specify number of times to run a model with different seeds
verbose | If True it prints stuff regarding the progress of an algorithm
threads | Int value to apply parallelism. Not always applicable, but can facilitate speed’s performance
usescale | If True it use maximum absolute scaling. It is useful for linear algorithms
copy | If True, it makes a hard copy of the data. 

## Classifiers

Classifier Models are described first.

### DecisionTreeClassifier

```
DecisionTreeClassifier threads:50 max_tree_size:-1 rounding=10 offset:0.0001 feature_subselection:1.0 cut_off_subsample:1.0 max_depth:6 max_features:0.9 min_leaf:5.0 min_split:10 Objective:ENTROPY row_subsample:0.95 seed:1 threads:1 bags:1 verbose:false
```

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

```
RandomForestClassifier bootsrap:false max_tree_size:-1 cut_off_subsample:1.0 feature_subselection:1.0 rounding:6 estimators:100 offset:0.00001 max_depth:6 max_features:0.4 min_leaf:2.0 min_split:5.0 Objective:ENTROPY row_subsample:0.95 seed:1 threads:1 bags:1 verbose:false
```

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

```
AdaboostRandomForestClassifier bootsrap:false trees:1 offset:0.00001 max_tree_size:-1 cut_off_subsample:1.0 weight_thresold:0.95 estimators:100 max_depth:6 max_features:0.5 min_leaf:2.0 min_split:5.0 Objective:ENTROPY row_subsample:0.9 seed:1 threads:1 bags:1 verbose:false
```

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

```
GradientBoostingForestClassifier rounding:6 estimators:1000 shrinkage:0.1 offset:0.00001 max_tree_size:-1 cut_off_subsample:1.0 max_depth:8 max_features:0.4 min_leaf:4.0 min_split:8.0 Objective:RMSE row_subsample:0.7 seed:1 threads:1 bags:1 verbose:false
```

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

```
LogisticRegression Type:Liblinear C:1.0 l1C:1.0 learn_rate:0.1 shuffle:true RegularizationType:L2 UseConstant:true usescale:True maxim_Iteration:200 seed:1 threads:1 bags:1 verbose:false
```

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

```
LSVC Type:Liblinear usescale:True C:1.0 RegularizationType:L2 shuffle:true UseConstant:true l1C:1.0 maxim_Iteration:100 seed:1 threads:1 bags:1 verbose:false
```

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

```
LibFmClassifier maxim_Iteration:50 C:0.001 C2:0.001 shuffle:true lfeatures:2 UseConstant:true usescale:True init_values:0.1 learn_rate:0.1 smooth:0.01 seed:1 threads:1 bags:1 verbose:false
```

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

```
softmaxnnclassifier usescale:True maxim_Iteration:50 UseConstant:true C:0.000001 shuffle:true tolerance:0.01 learn_rate:0.01 smooth:0.1 h1:20 h2:20 connection_nonlinearity:Relu init_values:0.02 seed:1 threads:1 bags:1 verbose:false
```

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
```
NaiveBayesClassifier usescale:True Shrinkage:0.1 seed:1 threads:1 verbose:false
```

Parameter | Explanation
--- | ---
Shrinkage | Can be seen as a form of a penalty to avoid really big product’s failures. 

### XgboostClassifier

The original parameters can be found [here](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md) 

```
XgboostClassifier booster:gbtree num_round:1000 eta:0.005 max_leaves:0 gamma:1. max_depth:5 min_child_weight:1.0 subsample:0.9 colsample_bytree:0.7 colsample_bylevel:1.0 lambda:1.0 alpha:1.0 seed:1 threads:1 bags:1 verbose:false
```

Parameter | Explanation
--- | ---
scale_pos_weight | used for imbalanced classes(double)
num_round | Number of estimators to build (int) . **This is important**.
max_leaves |Maximum leaves in a tree (int).
eta | Penalty applied to each estimator. Needs to be between 0 and 1 (double). **This is important**.
max_depth | Maximum depth of the tree (int). **This is important**.
subsample | Proportion of observations to consider (double). **This is important**.
colsample_bylevel | Proportion of columns (features) to consider in each level (double).
colsample_bytree |  Proportion of columns (features) to consider in each Tree (double) **This is important**.
max_delta_step | controls optimization step (double).
gamma | controls minimum change requirements in loss to allow for a split (double).
booster | 'gbtree' or 'gblinear'.
alpha |controls overfitting (double). 
lambda |controls overfitting (double). 

### LightgbmClassifier

The original parameters can be found [here](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.md) 

```
LightgbmClassifier boosting:gbdt num_leaves:14 num_iterations:100 scale_pos_weight:1.0 skip_drop:0.5 uniform_drop:false xgboost_dart_mode:false two_round:false top_rate:0.1 sigmoid:1.0 is_unbalance:false max_bin:255 poission_max_delta_step:0.7 min_sum_hessian_in_leaf:0.0001 other_rate:0.1 min_data_in_bin:5 max_drop:50 drop_rate:0.1 categorical_feature:0,1,2 learning_rate:0.1 threads:1 max_depth:5 feature_fraction:0.5 min_data_in_leaf:10 min_gain_to_split:20 bagging_fraction:0.9 lambda_l1:0.1 lambda_l2:0.1 bagging_freq:1 bin_construct_sample_cnt:100000 seed:1 threads:1 bags:1 verbose:false
```

Parameter | Explanation
--- | ---
learning_rate | weight of each estimator. **This is important**
bagging_fraction | Proportions of rows consider. **This is important**
num_iterations | Number of trees to build. **This is important**
max_depth | maximum depth of the tree. **This is important**
feature_fraction | Proportions of columns (features) to consider within a tree. **This is important**
bagging_freq | Every how many iters it will perform bagging.
bin_construct_sample_cnt | Sample number of rows to create histograms.
boosting | Type of boosting. Could be 'gbdt','dart' or 'goss' . 
categorical_feature | comma separated features to be treated as categorical
drop_rate | dropout rate in dart boosting
is_unbalance | true to oversample weak classes in binary classification
lambda_l1 | L1 regularization
lambda_l2 | L2 regularization
max_bin |  max number of bin that feature values will bucket in.
max_drop | max number of dropped trees on one iteration (in dart).
min_data_in_bin | min number of data inside one bin, use this to avoid one-data-one-bin (may prevent over-fitting).
min_data_in_leaf | Minimum number of data in a leaf.
min_gain_to_split | Minimum gain to split a node
min_sum_hessian_in_leaf | Minimum sum hessian in one leaf
num_leaves | maximum number of leaves.
other_rate | only used in boosting goss, the retain ratio of small gradient data.
poission_max_delta_step | safeguard optimisation.
scale_pos_weight | scale weight for binary class.
sigmoid | parameter for sigmoid function.
skip_drop | probability of skipping drop (in dart).
top_rate | used in boosting goss, the retain ratio of large gradient data.
two_round | if true it saves memory but takes more time. 
uniform_drop | Specify whether to use uniform dropout.
boolean	xgboost_dart_mode | true use xgboost dart mode or not.

### SklearnAdaBoostClassifier

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) 

```
SklearnAdaBoostClassifier algorithm:SAMME.R learning_rate:0.7 n_estimators:100 threads:1 usedense:false seed:1 verbose:false
```

Parameter | Explanation
--- | ---
learning_rate | Learning rate shrinks the contribution of each classifier by learning_rate. **This is important**
n_estimators | Number of trees to build. **This is important**
algorithm | Could be SAMME or SAMME.R
use_dense | True to Use dense data.

### SklearnDecisionTreeClassifier

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) 

```
SklearnDecisionTreeClassifier criterion:entropy max_leaf_nodes:0 min_impurity_split:0.0000001 threads:1 max_features:0.5 max_depth:5 min_samples_leaf:1 min_samples_split:2 use_dense:false min_weight_fraction_leaf:0.0 new_tree_gain_ratio:1.0 bootsrap:false seed:1 verbose:false
```

Parameter | Explanation
--- | ---
max_depth | maximum depth of the tree. **This is important**
max_features | Proportions of columns (features) to consider. **This is important**
max_leaf_nodes | maximum number of nodes allowed.
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 
min_impurity_split | Threshold for early stopping in tree growth.
criterion | Criterion to determine the split could gini or entropy
min_samples_leaf |Minimum cases to keep a splitted node
min_samples_split | Minimum cases to split a node
min_weight_fraction_leaf | The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.

### SklearnExtraTreesClassifier

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) 

```
SklearnExtraTreesClassifier criterion:entropy max_leaf_nodes:0 n_estimators:100 min_impurity_split:0.0000001 threads:1 max_features:0.5 max_depth:5 min_samples_leaf:1 min_samples_split:2 use_dense:false min_weight_fraction_leaf:0.0 new_tree_gain_ratio:1.0 bootsrap:false seed:1 verbose:false 		  
```

Parameter | Explanation
--- | ---
n_estimators | Toral number of trees to build. **This is important**
max_depth | maximum depth of the tree. **This is important**
max_features | Proportions of columns (features) to consider. **This is important**
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 
max_leaf_nodes | maximum number of nodes allowed.
min_impurity_split | Threshold for early stopping in tree growth.
bootsrap | true use bootsrap or not.
criterion | Criterion to determine the split could gini or entropy
min_samples_leaf |Minimum cases to keep a splitted node
min_samples_split | Minimum cases to split a node
min_weight_fraction_leaf | The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.

### SklearnRandomForestClassifier

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.SklearnRandomForestClassifier.html) 

```
SklearnRandomForestClassifier criterion:entropy max_leaf_nodes:0 n_estimators:100 min_impurity_split:0.0000001 threads:1 max_features:0.5 max_depth:5 min_samples_leaf:1 min_samples_split:2 use_dense:false min_weight_fraction_leaf:0.0 new_tree_gain_ratio:1.0 bootsrap:false seed:1 verbose:false 		  
```

Parameter | Explanation
--- | ---
n_estimators | Toral number of trees to build. **This is important**
max_depth | maximum depth of the tree. **This is important**
max_features | Proportions of columns (features) to consider. **This is important**
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 
max_leaf_nodes | maximum number of nodes allowed.
min_impurity_split | Threshold for early stopping in tree growth.
bootsrap | true use bootsrap or not.
criterion | Criterion to determine the split could gini or entropy
min_samples_leaf |Minimum cases to keep a splitted node
min_samples_split | Minimum cases to split a node
min_weight_fraction_leaf | The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.

### SklearnMLPClassifier

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier) 

```
SklearnMLPClassifier standardize:true use_log1p:true shuffle:true learning_rate:adaptive momentum:0.9 optimizer:sgd use_dense:false alpha:0.000001 hidden:50,50 activation:relu epsilon:0.00000001 max_iter:50 learning_rate_init:0.01 batch_size:8 tol:0.0001 validation_split:0.2 seed:1 verbose:false
```

Parameter | Explanation
--- | ---
hidden | Toral Comma-separated integers defining the number of hidden layers along with teh hidden units. **This is important**
epochs | Maximum number of iterations. **This is important**
activation | Activation function for the hidden layer coudl be identity, logistic, tanh, relu. **This is important**
alpha | L2 regularization on the weights. **This is important**
learning_rate_init | The (initial) learning rate used. **This is important**
learning_rate | Could be adaptive ,constant or invscaling.	
batch_size |  Number of cases(samples) in a batch.
optimizer | could adam, bfgs or sgd.
tol | Tolerance to determine the end of the optimization.
epsilon | Value for numerical stability in adam.
momentum | Only applicable for optimizer=sgd. Nesterov's is on by default.
shuffle | true Enable shuffling of training data (on each epoc).
standardize | true to standardize dense data (e.g use_dense=true) or to use absmaxscaling on sparse data (use_dense=false).
use_log1p | converts the data matrix to log plus 1.
validation_split | Split percentage to use for early stopping. Best epoc is determived via cv after 2 consecutive worse loss estimateds.
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 

### SklearnSGDClassifier

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier) 

```
SklearnSGDClassifier standardize:true use_log1p:true shuffle:true learning_rate:adaptive momentum:0.9 optimizer:sgd use_dense:false alpha:0.000001 hidden:50,50 activation:relu epsilon:0.00000001 max_iter:50 learning_rate_init:0.01 batch_size:8 tol:0.0001 validation_split:0.2 seed:1 verbose:false
```

Parameter | Explanation
--- | ---
n_iter | Maximum number of iterations. **This is important**
alpha | Regularization on the weights. **This is important**
eta0 | The (initial) learning rate used. **This is important**
learning_rate | Could be optimal, constant or invscaling.	
loss | could be log or modified_huber.
epsilon | For huber, determines the threshold at which it becomes less important to get the prediction exactly right.
l1_ratio | The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
penalty | The penalty (aka regularization term) to be used. could be l2, l1, or elasticnet .
power_t | The exponent for inverse scaling learning rate [default 0.5].
shuffle | true Enable shuffling of training data (on each iteration).
standardize | true to standardize dense data (e.g use_dense=true) or to use absmaxscaling on sparse data (use_dense=false).
use_log1p | converts the data matrix to log plus 1.
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 

### SklearnknnClassifier

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) 

```
SklearnknnClassifier seed:1 usedense:true use_scale:false distance:cityblock metric:uniform n_neighbors:3 thread:1 verbose:false
```

Parameter | Explanation
--- | ---
n_neighbors |  Number of neighbors to use by default for k_neighbors queries. **This is important**
distance | It must be one of euclidean, cosine, manhattan or cityblock
metric | Weight function used in prediction. Possible values: uniform or distance.
use_scale | true to use absmaxscaling.
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 

### SklearnsvmClassifier

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) 

```
SklearnsvmClassifier seed:1 usedense:false use_scale:false max_iter:-1 kernel:rbf degree:3 C:1.0 tol:0.0001 coef0:0.0 gamma:0.0 verbose:False
```

Parameter | Explanation
--- | ---
max_iter | Maximum number of iterations. **This is important**
kernel | Kernel type could be linear, poly, rbf or sigmoid. **This is important**
C | The Penalty parameter C of the error term. **This is important**
tol | Tolerance to determine the end of the optimization.	
degree | Degree of the polynomial kernel function (poly).	
gamma | Kernel coefficient for rbf, poly and sigmoid.
coef | Independent term in kernel function.It is only significant in poly and sigmoid.
use_scale | true to use absmaxscaling.
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 

### KerasnnClassifier

The original parameters can be found sparsely in [keras' documentation](https://keras.io/) 

```
KerasnnClassifier loss:categorical_crossentropy standardize:true use_log1p:true shuffle:true batch_normalization:true weight_init:lecun_uniform momentum:0.9 optimizer:adam use_dense:true l2:0.000001,0.000001 hidden:50,50 activation:relu,relu droupouts:0.1,0.1 epochs:20 lr:0.01 batch_size:8 stopping_rounds:10 validation_split:0.2 seed:1 verbose:false 
```

Parameter | Explanation
--- | ---
hidden | Toral Comma-separated integers defining the number of hidden layers along with teh hidden units. **This is important**
droupouts | Toral Comma-separated floats defining the dropout in each layer (defined by hidden). **This is important**
l2 | Toral Comma-separated floats defining the l2 regularization term on the weights in each layer (defined by hidden). **This is important**
activation | Toral Comma-separated strings defining the activation in each hidden layer. **This is important**
lr | The learning rate used. **This is important**
epochs | Maximum number of iterations. **This is important**
batch_normalization | true to add a batch normlization to the layers. **This is important**
batch_size |  Number of cases(samples) in a batch. **This is important**
weight_init |  The distribution from which initial weights are to be drawn. Has to be RandomNormal, RandomUniform, TruncatedNormal, VarianceScaling, Orthogonal, Identity, lecun_uniform, glorot_normal,glorot_uniform, he_normal, lecun_normal, he_uniform, he_normal.
optimizer | Has to be adam, adagrad, nadam, adadelta or sgd.	
loss | Has to be categorical_crossentropy, categorical_hinge, logcosh, Kullback–Leibler divergence.	 
momentum | Only applicable for optimizer=sgd. Nesterov's is on by default.
shuffle | true Enable shuffling of training data (on each epoc).
standardize | true to standardize dense data (e.g use_dense=true) or to use absmaxscaling on sparse data (use_dense=false).
use_log1p | converts the data matrix to log plus 1.
validation_split | Split percentage to use for early stopping. Best epoc is determived via cv after 2 consecutive worse loss estimateds.
stopping_rounds | Split percentage to use for early stopping. Best epoc is determived via cv after 2 consecutive worse loss estimateds.
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 

### PythonGenericClassifier

The user can run his/her own python script as long as it is placed in **lib/python/** and named after  PythonGenericClassifier[INDEX]. Index will be a hyper parameter. Look for PythonGenericClassifier0.py in lib/python/ for an example.

```
PythonGenericClassifier index:0 seed:1 verbose:False 
```

Parameter | Explanation
--- | ---
index |  this is the index specifying which PythonGenericClassifier[index].py script to run. **This is important**


### FRGFClassifier

(Some of) the original parameters of [fast_rgf](https://github.com/baidu/fast_rgf) can be found [here](https://github.com/baidu/fast_rgf/tree/master/examples) 

```
FRGFClassifier dtree_loss:LOGISTIC max_nodes:16 ntrees:100 eta:0.1 threads:1 sparse_max_features:1000 max_level:4 dense_max_buckets:250 min_sample:1 min_occurrences:1 opt:rgf sparse_lamL2:1.0 min_bucket_weights:250 new_tree_gain_ratio:1.0 stepsize:0.1 lamL2:1.0 lamL1:1.0 sparse_max_buckets:250 verbose:False
```

Parameter | Explanation
--- | ---
ntrees | Toral number of trees to build. **This is important**
max_level | maximum depth of the tree. **This is important**
lamL2 | L2 regularization on the weights. **This is important**
new_tree_gain_ratio | new tree is created when leaf-nodes gain < this value * estimated gain of creating new three. **This is important**
lamL1 | L1 regularization on the weights.
stepsize | Step size of epsilon-greedy boosting (inactive for rgf).
min_occurrences | minimum number of occurrences for a feature to be selected.
min_sample | minimum samples in node.
max_nodes | maximum number of nodes.
loss | Type of loss. could be LS, MODLS (modified least squares loss), or LOGISTIC.
opt | optimization method for training forest. Could be rgf or epsilon-greedy.
sparse_lamL2 | L2 regularization parameter for sparse data.
min_bucket_weights | Minimum sum of data weights for each discretized value.
dense_max_buckets | Maximum bins for dense data.
sparse_max_features | You may try a different value in [1000,10000000] for fetaures allowed.
dense_max_buckets | Maximum bins for dense data.

### H2OGbmClassifier

```
H2OGbmClassifier ntrees:100 learn_rate:0.01 nbins:255 balance_classes:false max_depth:4 col_sample_rate_per_tree:0.5 col_sample_rate:1.0 sample_rate:0.9 min_rows:1 seed:1 threads:1 bags:1 verbose:false
```

Parameter | Explanation
--- | ---
col_sample_rate | Proportions of columns (features) to consider at each level of a given tree. **This is important**
learn_rate | weight on each estimator. **This is important**
max_depth | maximum depth of the tree. **This is important**
ntrees | Number of trees to build **This is important**
sample_rate | Proportions of rows consider **This is important**
col_sample_rate_per_tree | Proportions of columns (features) to consider within a tree.
balance_classes | whether to oversample the minority classes to balance the class distribution.
min_rows | minimum number of cases in a node.
nbins | The number of bins for the histogram to build.

### H2ODeepLearningClassifier

```
H2ODeepLearningClassifier activation:Rectifier input_dropout_ratio:0.1 shuffle:true tandardize:false weight_init:UniformAdaptive sample_rate:1.0 l1:0 l2:0.00001 max_w2:1.0 mini_batch_size:1 fast_mode:false adaptive_rate:true rho:0.9 epsilon:1e-8 balance_classes:false epochs:10 dropouts:0.5,0.5 hidden:100,50 col_sample_rate:1.0 sample_rate:0.9 min_rows:1 seed:1 threads:1 bags:1 verbose:false
```

Parameter | Explanation
--- | ---
activation | activation functions. Has to be between 'Rectifier', 'Tanh', 'ExpRectifier' or 'Maxout'
adaptive_rate | true to use The implemented adaptive learning rate algorithm (ADADELTA) which automatically combines the benefits of learning rate annealing and momentum training to avoid slow convergence.
rho | The first of two hyper parameters for ADADELTA. It is like momentum. **This is important**
epsilon | The second of two hyper parameters for ADADELTA. **This is important**
balance_classes | Specify whether to oversample the minority classes to balance the class distribution.
dropouts | dropout ratios for each hidden layer,comma separated .Has to match in length the 'hidden' parameter. **This is important**
epochs | Number of iterations to train the DL model. **This is important**
fast_mode | True for faster convergence (but potential loss in accuracy)
hidden | Number of hidden neurons, comma separated.The length connotes the number of hidden layers too. **This is important**
input_dropout_ratio | dropout from to the input layer
l1 | regularization on the weights.
l2 | regularization on the weights. **This is important**
max_w2 | A maximum on the sum of the squared incoming weights into any one neuron.
mini_batch_size | minimum number of cases in batch.
momentum_ramp | The momentum_ramp parameter controls the amount of learning for which momentum increases (assuming momentum_stable is larger than momentum_start).
momentum_stable | The momentum_stable parameter controls the final momentum value reached after momentum_ramp training samples.
momentum_start | The momentum_start parameter controls the amount of momentum at the beginning of training.
nesterov_accelerated_gradient | True to enable Nesterov accelerated gradient descent method.
rate | When adaptive learning rate is disabled, the magnitude of the weight updates are determined by the user specified learning rate (potentially annealed), and are a function of the difference between the predicted value and the target value.
rate_annealing | Learning rate annealing reduces the learning rate to “freeze” into local minima in the optimization landscape.
rate_decay | The learning rate decay parameter controls the change of learning rate across layers.
sample_rate | Proportions of rows consider in each epoc.
shuffle | true to enable shuffling of training data (on each node).
tandardize | true to standardize the input data.
weight_init | The distribution from which initial weights are to be drawn. Has to be 'UniformAdaptive', 'Uniform' or 'Normal'

### H2ODrfClassifier

```
H2ODrfClassifier ntrees:100 nbins:255 balance_classes:false max_depth:4 col_sample_rate_per_tree:0.5 sample_rate:0.9 min_rows:1 seed:1 threads:1 bags:1 verbose:false
```

Parameter | Explanation
--- | ---
max_depth | maximum depth of the tree. **This is important**
ntrees | Number of trees to build. **This is important**
sample_rate | Proportions of rows consider **This is important**
col_sample_rate_per_tree | Proportions of columns (features) to consider within a tree.
balance_classes | whether to oversample the minority classes to balance the class distribution.
min_rows | minimum number of cases in a node.
nbins | The number of bins for the histogram to build.

### H2OGlmClassifier

```
H2OGlmClassifier alpha:0 lambda:0.00001 balance_classes:false standardize:false max_iterations:50 beta_epsilon:0.00001 bjective_epsilon:0.00001 seed:1 threads:1 bags:1 verbose:false
```

Parameter | Explanation
--- | ---
alpha | Proportion of l1/l2. 0 = Ridge, 1=Lasso
lambda | Regularization parameter. **This is important**
max_iterations | Number of iterations to build the model. **This is important**
beta_epsilon | tolerance of the coefficients
bjective_epsilon | tolerance of the objective function
balance_classes | true to Specify whether to oversample the minority classes to balance the class distribution.
standardize | true to standardize input features or not

### OriginalLibFMClassifier

Wraps the original implementation of [ibFM](http://libfm.org/), made from Steffen Rendle. The reason this implementation is made, is because internal results show that it has better performance (as in accuracy) than StackNet's internal implementation and has more training methods than just sgd. 
This implementation may not include all libFM features plus it actually uses a version of it **that had a bug(!)** on purpose. You can find more information about why this was chosen in the following [python wrapper for libFM](https://github.com/jfloff/pywFM). It basically had this bug that was allowing you to get the parameters of the trained models for all training methods. These parameters are now extracted once a model has been trained and the scoring uses only these parameters (e.g. not the libFM executable). 
Don't forget to acknowledge libFM if you publish results produced with this software. Also take note of its licence GNU. More information can be found on [libFM's repo on github](https://github.com/srendle/libfm).

```
OriginalLibFMClassifier type:als lfeatures:4 learn_rate:0.01 maxim_Iteration:10 init_values:0.01 c:1 c2:1 threads:1 usescale:false seed:1 verbose:true bags:1 
```

Parameter | Explanation
--- | ---
Type | Type of algorithm to use. It has to be  sgd, als, mcmc. Default is mcmc.
C | Regularization value, the more, the stronger the regularization. **This is important**
C2 | Regularization value for the latent features. **This is important**
lfeatures | Number of latent features to use. **This is important**
init_values | Initialise values of the latent features with values between[0,init_values). **This is important**
maxim_Iteration | aximum number of iterations. **This is important**
learn_rate | learn_rate for SGD; default=0.1. **This is important**

### VowpaLWabbitClassifier

Wrapper for [vowpal wabbit](https://github.com/JohnLangford/vowpal_wabbit/wiki). It does not contain all features, but a fraction.
```
VowpaLWabbitClassifier use_ftrl:True learning_rate:0.8 decay_learning_rate:0.97 nn:40 bit_precision:18 ftrl_alpha:0.1 ftrl_beta:0.1 power_t:0.9 initial_t:0.9 l1:0.01 l2:0.01 passes:10 threads:1 make2way:false make3way:false use_dropout:true use_meanfield:false seed:1 verbose:true bags:1  
```

Parameter | Explanation
--- | ---
passes | Number of training Passes. **This is important**
bit_precision | number of bits in the feature table. 
decay_learning_rate | Decay factor for learning_rate between passes. 
nn | Number of hidden units to use in a sigmoidal feedforward network with nn hidden units
initial_t | Initial t value. Affects learning rate's updates
power_t | t power value. Affects learning rate's updates
ftrl_alpha | ftrl alpha parameter when using ftrl **This is important**
ftrl_beta | ftrl beta stability patameter when using ftrl **This is important**
learning_rate | learning Rate for gradient-based updates
l1 | L1 regularization 
l2 | L2 regularization **This is important**
use_ftrl | true to use the ftrl optimization option (instead of adaptive). It is on by default.
make2way | if true it creates all possible 2-way interactions of all features
make3way | if true it creates all possible 3-way interactions of all features
use_dropout | when nn>0, train or test sigmoidal feedforward network using dropout.
use_meanfield | when nn>0, train or test sigmoidal feedforward network using mean field.

### libffmClassifier

Wraps [Libffm](https://github.com/guestwalk/libffm). Note this method either requires the user to manually add comma separated indices that form a field or they need to use some self-made heuristics. This is controlled by parameter `opt`.

```
libffmClassifier factor:6 iteration:16 learn_rate:0.1 opt:order lambda:0.0001 threads:1 use_norm:false seed:1 verbose:true bags:1  
```

Parameter | Explanation
--- | ---
factor | number of latent factors. **This is important**
iteration | number of iterations. **This is important** 
learn_rate | learning rate. **This is important**  
lambda |regularization parameter. **This is important**  
use_norm | true to allow instance-wise normalization. **This is important**
opt | method for determining the factors. The best way (but not the default) is to provide a list with comma separated indices. Consider this String '1,4,7,123,546'. This would mean that the 0 column is a field on its own, {1,2,3} form another field, {4,5,6} another. {7,8...122} form another field and so on. Another possible value is 'no_order' (default). This looks at the proportion of zeros in neighbouring columns to determine if they form a field. The last possible value is 'order'. This calculates frequencies of non-zero values for all columns and then orders them based on frequency. Columns that have a few missing values form their own fields. Weaker columns (frequency-wise) are joined together to form fields. 


## Regressors

### DecisionTreeRegressor

```
DecisionTreeRegressor threads:50 max_tree_size:-1 rounding=10 offset:0.0001 feature_subselection:1.0 cut_off_subsample:1.0 max_depth:6 max_features:0.9 min_leaf:5.0 min_split:10 Objective:RMSE row_subsample:0.95 seed:1 threads:1 bags:1 verbose:false
```

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

```
RandomForestRegressor bootsrap:false max_tree_size:-1 cut_off_subsample:1.0 feature_subselection:1.0 rounding:6 estimators:100 offset:0.00001 max_depth:6 max_features:0.4 min_leaf:2.0 min_split:5.0 Objective:RMSE row_subsample:0.95 seed:1 threads:1 bags:1 verbose:false
```

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

```
AdaboostRandomForestRegressor bootsrap:false trees:1 offset:0.00001 max_tree_size:-1 cut_off_subsample:1.0 weight_thresold:0.95 estimators:100 max_depth:6 max_features:0.5 min_leaf:2.0 min_split:5.0 Objective:RMSE row_subsample:0.9 seed:1 threads:1 bags:1 verbose:false
```

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

```
GradientBoostingForestRegressor rounding:6 estimators:1000 shrinkage:0.1 offset:0.00001 max_tree_size:-1 cut_off_subsample:1.0 max_depth:8 max_features:0.4 min_leaf:4.0 min_split:8.0 Objective:RMSE row_subsample:0.7 seed:1 threads:1 bags:1 verbose:false
```

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

```
LinearRegression Type:Routine C:1.0 l1C:1.0 learn_rate:0.1 Objective:RMSE tau:0.5 shuffle:true RegularizationType:L2 UseConstant:true usescale:True maxim_Iteration:200 seed:1 threads:1 bags:1 verbose:false
```

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

```
LSVR Type:Liblinear usescale:True C:1.0 learn_rate:0.1 smooth:0.1 RegularizationType:L2 Objective:L2 shuffle:true UseConstant:true l1C:1.0 maxim_Iteration:100 seed:1 threads:1 bags:1 verbose:false
```

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

```
LibFmRegressor maxim_Iteration:50 C:0.001 Objective:“RMSE” tau:0.5 C2:0.001 shuffle:true lfeatures:2 UseConstant:true usescale:True init_values:0.1 learn_rate:0.1 smooth:0.01 seed:1 threads:1 bags:1 verbose:false
```

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

```
multinnregressor usescale:True maxim_Iteration:50 Objective:RMSE tau:0.5 UseConstant:true C:0.000001 shuffle:true tolerance:0.01 learn_rate:0.01 smooth:0.1 h1:20 h2:20 connection_nonlinearity:Relu init_values:0.02 seed:1 threads:1 bags:1 verbose:false
```

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
UseConstant | If true it uses an intercept.
shuffle | True to train on random rows.

### XgboostRegressor

The original parameters can be found [here](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md) 

```
XgboostRegressor booster:gbtree num_round:1000 eta:0.005 max_leaves:0 gamma:1. max_depth:5 min_child_weight:1.0 subsample:0.9 colsample_bytree:0.7 colsample_bylevel:1.0 lambda:1.0 alpha:1.0 seed:1 threads:1 bags:1 verbose:false
```

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

### LightgbmRegressor

The original parameters can be found [here](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.md) 

```
LightgbmRegressor boosting:gbdt objective:regression huber_delta:0.1 fair_c:0.1 num_leaves:14 num_iterations:100 scale_pos_weight:1.0 skip_drop:0.5 uniform_drop:false xgboost_dart_mode:false two_round:false top_rate:0.1 sigmoid:1.0 is_unbalance:false max_bin:255 poission_max_delta_step:0.7 min_sum_hessian_in_leaf:0.0001 other_rate:0.1 min_data_in_bin:5 max_drop:50 drop_rate:0.1 categorical_feature:0,1,2 learning_rate:0.1 threads:1 max_depth:5 feature_fraction:0.5 min_data_in_leaf:10 min_gain_to_split:20 bagging_fraction:0.9 lambda_l1:0.1 lambda_l2:0.1 bagging_freq:1 bin_construct_sample_cnt:100000 seed:1 threads:1 bags:1 verbose:false
```

Parameter | Explanation
--- | ---
learning_rate | weight of each estimator. **This is important**
bagging_fraction | Proportions of rows consider. **This is important**
num_iterations | Number of trees to build. **This is important**
max_depth | maximum depth of the tree. **This is important**
feature_fraction | Proportions of columns (features) to consider within a tree. **This is important**
objective | has to be 'regression','regression_l1','fair' ,'huber','poisson'
huber_delta | parameter for Huber loss. Will be used in regression task.	
fair_c | parameter for Fair loss. Will be used in regression task.
bagging_freq | Every how many iters it will perform bagging.
bin_construct_sample_cnt | Sample number of rows to create histograms.
boosting | Type of boosting. Could be 'gbdt','dart' or 'goss' . 
categorical_feature | comma separated features to be treated as categorical
drop_rate | dropout rate in dart boosting
is_unbalance | true to oversample weak classes in binary classification
lambda_l1 | L1 regularization
lambda_l2 | L2 regularization
max_bin |  max number of bin that feature values will bucket in.
max_drop | max number of dropped trees on one iteration (in dart).
min_data_in_bin | min number of data inside one bin, use this to avoid one-data-one-bin (may prevent over-fitting).
min_data_in_leaf | Minimum number of data in a leaf.
min_gain_to_split | Minimum gain to split a node
min_sum_hessian_in_leaf | Minimum sum hessian in one leaf
num_leaves | maximum number of leaves.
other_rate | only used in boosting goss, the retain ratio of small gradient data.
poission_max_delta_step | safeguard optimisation.
scale_pos_weight | scale weight for binary class.
sigmoid | parameter for sigmoid function.
skip_drop | probability of skipping drop (in dart).
top_rate | used in boosting goss, the retain ratio of large gradient data.
two_round | if true it saves memory but takes more time. 
uniform_drop | Specify whether to use uniform dropout.
boolean	xgboost_dart_mode | true use xgboost dart mode or not.

### H2OGbmRegressor

```
H2OGbmRegressor ntrees:100 tweedie_power:1.2 quantile_alpha:0.1 objective:auto learn_rate:0.01 nbins:255 balance_classes:false max_depth:4 col_sample_rate_per_tree:0.5 col_sample_rate:1.0 sample_rate:0.9 min_rows:1 seed:1 threads:1 bags:1 verbose:false
```

Parameter | Explanation
--- | ---
col_sample_rate | Proportions of columns (features) to consider at each level of a given tree. **This is important**
learn_rate | weight on each estimator. **This is important**
max_depth | maximum depth of the tree. **This is important**
ntrees | Number of trees to build **This is important**
sample_rate | Proportions of rows consider **This is important**
col_sample_rate_per_tree | Proportions of columns (features) to consider within a tree.
balance_classes | whether to oversample the minority classes to balance the class distribution.
min_rows | minimum number of cases in a node.
nbins | The number of bins for the histogram to build.
tweedie_power | Only applicable if Tweedie is specified for distribution) Specify the Tweedie power. The range is from 1 to 2. For a normal distribution, enter 0. For Poisson distribution, enter 1. For a gamma distribution, enter 2. For a compound Poisson-gamma distribution, enter a value greater than 1 but less than 2.
quantile_alpha | Only applicable if Quantile is specified for distribution) Specify the quantile to be used for Quantile Regression.
objective | The objective has to be one of [auto, gamma gaussian huber laplace  poisson quantile tweedie].

### H2ODeepLearningRegressor

```
H2ODeepLearningRegressor activation:Rectifier tweedie_power:1.2 quantile_alpha:0.1 objective:auto loss:Automatic input_dropout_ratio:0.1 shuffle:true tandardize:false weight_init:UniformAdaptive sample_rate:1.0 l1:0 l2:0.00001 max_w2:1.0 mini_batch_size:1 fast_mode:false adaptive_rate:true rho:0.9 epsilon:1e-8 balance_classes:false epochs:10 dropouts:0.5,0.5 hidden:100,50 col_sample_rate:1.0 sample_rate:0.9 min_rows:1 seed:1 threads:1 bags:1 verbose:false
```

Parameter | Explanation
--- | ---
activation | activation functions. Has to be between 'Rectifier', 'Tanh', 'ExpRectifier' or 'Maxout'
adaptive_rate | true to use The implemented adaptive learning rate algorithm (ADADELTA) which automatically combines the benefits of learning rate annealing and momentum training to avoid slow convergence.
rho | The first of two hyper parameters for ADADELTA. It is like momentum. **This is important**
epsilon | The second of two hyper parameters for ADADELTA. **This is important**
balance_classes | Specify whether to oversample the minority classes to balance the class distribution.
dropouts | dropout ratios for each hidden layer,comma separated .Has to match in length the 'hidden' parameter. **This is important**
epochs | Number of iterations to train the DL model. **This is important**
fast_mode | True for faster convergence (but potential loss in accuracy)
hidden | Number of hidden neurons, comma separated.The length connotes the number of hidden layers too. **This is important**
input_dropout_ratio | dropout from to the input layer
l1 | regularization on the weights.
l2 | regularization on the weights. **This is important**
max_w2 | A maximum on the sum of the squared incoming weights into any one neuron.
mini_batch_size | minimum number of cases in batch.
momentum_ramp | The momentum_ramp parameter controls the amount of learning for which momentum increases (assuming momentum_stable is larger than momentum_start).
momentum_stable | The momentum_stable parameter controls the final momentum value reached after momentum_ramp training samples.
momentum_start | The momentum_start parameter controls the amount of momentum at the beginning of training.
nesterov_accelerated_gradient | True to enable Nesterov accelerated gradient descent method.
rate | When adaptive learning rate is disabled, the magnitude of the weight updates are determined by the user specified learning rate (potentially annealed), and are a function of the difference between the predicted value and the target value.
rate_annealing | Learning rate annealing reduces the learning rate to “freeze” into local minima in the optimization landscape.
rate_decay | The learning rate decay parameter controls the change of learning rate across layers.
sample_rate | Proportions of rows consider in each epoc.
shuffle | true to enable shuffling of training data (on each node).
tandardize | true to standardize the input data.
weight_init | The distribution from which initial weights are to be drawn. Has to be 'UniformAdaptive', 'Uniform' or 'Normal'
tweedie_power | Only applicable if Tweedie is specified for distribution) Specify the Tweedie power. The range is from 1 to 2. For a normal distribution, enter 0. For Poisson distribution, enter 1. For a gamma distribution, enter 2. For a compound Poisson-gamma distribution, enter a value greater than 1 but less than 2.
quantile_alpha | Only applicable if Quantile is specified for distribution) Specify the quantile to be used for Quantile Regression.
objective | The objective has to be of [auto, gamma ,gaussian ,huber ,laplace ,poisson ,quantile ,tweedie].
loss | The loss has to be one of [Automatic ,Absolute, Huber, Quadratic or Quantile]

### H2ODrfRegressor

```
H2ODrfRegressor ntrees:100 nbins:255 tweedie_power:1.2 quantile_alpha:0.1 objective:auto balance_classes:false max_depth:4 col_sample_rate_per_tree:0.5 sample_rate:0.9 min_rows:1 seed:1 threads:1 bags:1 verbose:false
```

Parameter | Explanation
--- | ---
max_depth | maximum depth of the tree. **This is important**
ntrees | Number of trees to build. **This is important**
sample_rate | Proportions of rows consider **This is important**
col_sample_rate_per_tree | Proportions of columns (features) to consider within a tree.
balance_classes | whether to oversample the minority classes to balance the class distribution.
min_rows | minimum number of cases in a node.
nbins | The number of bins for the histogram to build.
tweedie_power | Only applicable if Tweedie is specified for distribution) Specify the Tweedie power. The range is from 1 to 2. For a normal distribution, enter 0. For Poisson distribution, enter 1. For a gamma distribution, enter 2. For a compound Poisson-gamma distribution, enter a value greater than 1 but less than 2.
quantile_alpha | Only applicable if Quantile is specified for distribution) Specify the quantile to be used for Quantile Regression.
objective | The objective has to be one of [auto, ,gamma ,gaussian ,huber ,laplace ,poisson ,quantile ,tweedie].

### H2OGlmRegressor

```
H2OGlmRegressor alpha:0 lambda:0.00001 balance_classes:false standardize:false max_iterations:50 beta_epsilon:0.00001 bjective_epsilon:0.00001 seed:1 threads:1 bags:1 verbose:false
```

Parameter | Explanation
--- | ---
alpha | Proportion of l1/l2. 0 = Ridge, 1=Lasso
lambda | Regularization parameter. **This is important**
max_iterations | Number of iterations to build the model. **This is important**
beta_epsilon | tolerance of the coefficients
bjective_epsilon | tolerance of the objective function
balance_classes | true to Specify whether to oversample the minority classes to balance the class distribution.
standardize | true to standardize input features or not
tweedie_power | Only applicable if Tweedie is specified for distribution) Specify the Tweedie power. The range is from 1 to 2. For a normal distribution, enter 0. For Poisson distribution, enter 1. For a gamma distribution, enter 2. For a compound Poisson-gamma distribution, enter a value greater than 1 but less than 2.
quantile_alpha | Only applicable if Quantile is specified for distribution) Specify the quantile to be used for Quantile Regression.
family | The family has to be one of [auto, gamma ,gaussian ,poisson ,tweedie]
link | The link has to be one of [auto, log ,identity ,inverse ,tweedie]

### SklearnAdaBoostRegressor

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html) 

```
SklearnAdaBoostRegressor algorithm:square learning_rate:0.7 n_estimators:100 threads:1 usedense:false seed:1 verbose:false
```

Parameter | Explanation
--- | ---
learning_rate | Learning rate shrinks the contribution of each classifier by learning_rate. **This is important**
n_estimators | Number of trees to build. **This is important**
algorithm | Could be square, linear or exponential.
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 

### SklearnDecisionTreeRegressor

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) 

```
SklearnDecisionTreeRegressor criterion:mse max_leaf_nodes:0 min_impurity_split:0.0000001 threads:1 max_features:0.5 max_depth:5 min_samples_leaf:1 min_samples_split:2 use_dense:false min_weight_fraction_leaf:0.0 new_tree_gain_ratio:1.0 bootsrap:false seed:1 verbose:false
```

Parameter | Explanation
--- | ---
max_depth | maximum depth of the tree. **This is important**
max_features | Proportions of columns (features) to consider. **This is important**
max_leaf_nodes | maximum number of nodes allowed.
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 
min_impurity_split | Threshold for early stopping in tree growth.
criterion | Criterion to determine the split could be mse or mae
min_samples_leaf |Minimum cases to keep a splitted node
min_samples_split | Minimum cases to split a node
min_weight_fraction_leaf | The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.

### SklearnExtraTreesRegressor

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html) 

```
SklearnExtraTreesRegressor criterion:mse max_leaf_nodes:0 n_estimators:100 min_impurity_split:0.0000001 threads:1 max_features:0.5 max_depth:5 min_samples_leaf:1 min_samples_split:2 use_dense:false min_weight_fraction_leaf:0.0 new_tree_gain_ratio:1.0 bootsrap:false seed:1 verbose:false 		  
```

Parameter | Explanation
--- | ---
n_estimators | Toral number of trees to build. **This is important**
max_depth | maximum depth of the tree. **This is important**
max_features | Proportions of columns (features) to consider. **This is important**
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 
max_leaf_nodes | maximum number of nodes allowed.
min_impurity_split | Threshold for early stopping in tree growth.
bootsrap | true use bootsrap or not.
criterion | Criterion to determine the split could mse or mae
min_samples_leaf |Minimum cases to keep a splitted node
min_samples_split | Minimum cases to split a node
min_weight_fraction_leaf | The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.

### SklearnRandomForestRegressor

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) 

```
SklearnRandomForestRegressor criterion:mse max_leaf_nodes:0 n_estimators:100 min_impurity_split:0.0000001 threads:1 max_features:0.5 max_depth:5 min_samples_leaf:1 min_samples_split:2 use_dense:false min_weight_fraction_leaf:0.0 new_tree_gain_ratio:1.0 bootsrap:false seed:1 verbose:false 		  
```

Parameter | Explanation
--- | ---
n_estimators | Toral number of trees to build. **This is important**
max_depth | maximum depth of the tree. **This is important**
max_features | Proportions of columns (features) to consider. **This is important**
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 
max_leaf_nodes | maximum number of nodes allowed.
min_impurity_split | Threshold for early stopping in tree growth.
bootsrap | true use bootsrap or not.
criterion | Criterion to determine the split could mse or mae
min_samples_leaf |Minimum cases to keep a splitted node
min_samples_split | Minimum cases to split a node
min_weight_fraction_leaf | The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.

### SklearnMLPRegressor

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor) 

```
SklearnMLPRegressor standardize:true use_log1p:true shuffle:true learning_rate:adaptive momentum:0.9 optimizer:sgd use_dense:false alpha:0.000001 hidden:50,50 activation:relu epsilon:0.00000001 max_iter:50 learning_rate_init:0.01 batch_size:8 tol:0.0001 validation_split:0.2 seed:1 verbose:false
```

Parameter | Explanation
--- | ---
hidden | Toral Comma-separated integers defining the number of hidden layers along with teh hidden units. **This is important**
epochs | Maximum number of iterations. **This is important**
activation | Activation function for the hidden layer coudl be identity, logistic, tanh, relu. **This is important**
alpha | L2 regularization on the weights. **This is important**
learning_rate_init | The (initial) learning rate used. **This is important**
learning_rate | Could be adaptive ,constant or invscaling.	
batch_size |  Number of cases(samples) in a batch.
optimizer | could adam, bfgs or sgd.
tol | Tolerance to determine the end of the optimization.
momentum | Only applicable for optimizer=sgd. Nesterov's is on by default.
epsilon | Value for numerical stability in adam.
shuffle | true Enable shuffling of training data (on each epoc).
standardize | true to standardize dense data (e.g use_dense=true) or to use absmaxscaling on sparse data (use_dense=false).
use_log1p | converts the data matrix to log plus 1.
validation_split | Split percentage to use for early stopping. Best epoc is determived via cv after 2 consecutive worse loss estimateds.
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 

### SklearnSGDRegressor

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor) 

```
SklearnSGDRegressor standardize:true use_log1p:true shuffle:true learning_rate:constant l1_ratio:0.1 penalty:l2 use_dense:false alpha:0.00001 loss:squared_loss epsilon:0.00000001 n_iter:50 eta0:0.01 power_t:0.25 seed:1 threads:3 verbose:false
```

Parameter | Explanation
--- | ---
n_iter | Maximum number of iterations. **This is important**
alpha | Regularization on the weights. **This is important**
eta0 | The (initial) learning rate used. **This is important**
learning_rate | Could be optimal, constant or invscaling.	
loss | could be squared_loss, huber, epsilon_insensitive or squared_epsilon_insensitive.
epsilon |  For huber, determines the threshold at which it becomes less important to get the prediction right. For epsilon-insensitive, any differences between the current prediction and the correct label are ignored if they are less than this.
l1_ratio | The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
penalty | The penalty (aka regularization term) to be used. could be l2, l1, or elasticnet .
power_t | The exponent for inverse scaling learning rate [default 0.5].
shuffle | true Enable shuffling of training data (on each iteration).
standardize | true to standardize dense data (e.g use_dense=true) or to use absmaxscaling on sparse data (use_dense=false).
use_log1p | converts the data matrix to log plus 1.
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 


### SklearnknnRegressor

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) 

```
SklearnknnRegressor seed:1 usedense:true use_scale:false distance:cityblock metric:uniform n_neighbors:3 thread:1 verbose:false
```

Parameter | Explanation
--- | ---
n_neighbors |  Number of neighbors to use by default for k_neighbors queries. **This is important**
distance | It must be one of euclidean, cosine, manhattan or cityblock
metric | Weight function used in prediction. Possible values: uniform or distance.
use_scale | true to use absmaxscaling.
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 

### SklearnsvmRegressor

The original parameters can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) 

```
SklearnsvmRegressor seed:1 usedense:false use_scale:false max_iter:-1 kernel:rbf degree:3 C:1.0 tol:0.0001 coef0:0.0 gamma:0.0 verbose:False	
```

Parameter | Explanation
--- | ---
max_iter | Maximum number of iterations. **This is important**
kernel | Kernel type could be linear, poly, rbf or sigmoid. **This is important**
C | The Penalty parameter C of the error term. **This is important**
tol | Tolerance to determine the end of the optimization.	
degree | Degree of the polynomial kernel function (poly).	
gamma | Kernel coefficient for rbf, poly and sigmoid.
coef | Independent term in kernel function.It is only significant in poly and sigmoid.
use_scale | true to use absmaxscaling.
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 

### KerasnnRegressor

The original parameters can be found sparsely in [keras' documentation](https://keras.io/) 

```
KerasnnRegressor loss:mean_squared_error standardize:true use_log1p:true shuffle:true batch_normalization:true weight_init:lecun_uniform momentum:0.9 optimizer:adam use_dense:true l2:0.000001,0.000001 hidden:50,50 activation:relu,relu droupouts:0.1,0.1 epochs:20 lr:0.01 batch_size:8 stopping_rounds:10 validation_split:0.2 seed:1 verbose:false 
```

Parameter | Explanation
--- | ---
hidden | Toral Comma-separated integers defining the number of hidden layers along with teh hidden units. **This is important**
droupouts | Toral Comma-separated floats defining the dropout in each layer (defined by hidden). **This is important**
l2 | Toral Comma-separated floats defining the l2 regularization term on the weights in each layer (defined by hidden). **This is important**
activation | Toral Comma-separated strings defining the activation in each hidden layer. **This is important**
lr | The learning rate used. **This is important**
epochs | Maximum number of iterations. **This is important**
batch_normalization | true to add a batch normlization to the layers. **This is important**
batch_size |  Number of cases(samples) in a batch. **This is important**
weight_init |  The distribution from which initial weights are to be drawn. Has to be RandomNormal, RandomUniform, TruncatedNormal, VarianceScaling, Orthogonal, Identity, lecun_uniform, glorot_normal,glorot_uniform, he_normal, lecun_normal, he_uniform, he_normal.
optimizer | Has to be adam, adagrad, nadam, adadelta or sgd.	
loss | Has to be  mean_squared_error, mean_absolute_error, mean_squared_logarithmic_error, squared_hinge, hinge, poisson.	 
momentum | Only applicable for optimizer=sgd. Nesterov's is on by default.
shuffle | true Enable shuffling of training data (on each epoc).
standardize | true to standardize dense data (e.g use_dense=true) or to use absmaxscaling on sparse data (use_dense=false).
use_log1p | converts the data matrix to log plus 1.
validation_split | Split percentage to use for early stopping. Best epoc is determived via cv after 2 consecutive worse loss estimateds.
stopping_rounds | Split percentage to use for early stopping. Best epoc is determived via cv after 2 consecutive worse loss estimateds.
use_dense | True to Use dense data. If your data is in dense format, do select true as all files get loaded as sparse by default in python-based modules. 

### PythonGenericRegressor

The user can run his/her own python script as long as it is placed in **lib/python/** and named after PythonGenericRegressor[INDEX]. Index will be a hyper parameter. Look for PythonGenericRegressor0.py in lib/python/ for an example.

```
PythonGenericRegressor index:0 seed:1 verbose:False 
```

Parameter | Explanation
--- | ---
index |  this is the index specifying which PythonGenericRegressor[index].py script to run. **This is important**

### FRGFRegressor

(Some of) the original parameters of [fast_rgf](https://github.com/baidu/fast_rgf) can be found [here](https://github.com/baidu/fast_rgf/tree/master/examples) 

```
FRGFRegressor dtree_loss:LS max_nodes:16 ntrees:100 eta:0.1 threads:1 sparse_max_features:1000 max_level:4 dense_max_buckets:250 min_sample:1 min_occurrences:1 opt:rgf sparse_lamL2:1.0 min_bucket_weights:250 new_tree_gain_ratio:1.0 stepsize:0.1 lamL2:1.0 lamL1:1.0 sparse_max_buckets:250 verbose:False
```

Parameter | Explanation
--- | ---
ntrees | Toral number of trees to build. **This is important**
max_level | maximum depth of the tree. **This is important**
lamL2 | L2 regularization on the weights. **This is important**
new_tree_gain_ratio | new tree is created when leaf-nodes gain < this value * estimated gain of creating new three. **This is important**
lamL1 | L1 regularization on the weights.
stepsize | Step size of epsilon-greedy boosting (inactive for rgf).
min_occurrences | minimum number of occurrences for a feature to be selected.
min_sample | minimum samples in node.
max_nodes | maximum number of nodes.
loss | Type of loss. could be LS, MODLS (modified least squares loss), or LOGISTIC.
opt | optimization method for training forest. Could be rgf or epsilon-greedy.
sparse_lamL2 | L2 regularization parameter for sparse data.
min_bucket_weights | Minimum sum of data weights for each discretized value.
dense_max_buckets | Maximum bins for dense data.
sparse_max_features | You may try a different value in [1000,10000000] for fetaures allowed.
dense_max_buckets | Maximum bins for dense data.

### OriginalLibFMRegressor

Wraps the original implementation of [ibFM](http://libfm.org/), made from Steffen Rendle. The reason this implementation is made, is because internal results show that it has better performance (as in accuracy) than StackNet's internal implementation and has more training methods than just sgd. 
This implementation may not include all libFM features plus it actually uses a version of it **that had a bug(!)** on purpose. You can find more information about why this was chosen in the following [python wrapper for libFM](https://github.com/jfloff/pywFM). It basically had this bug that was allowing you to get the parameters of the trained models for all training methods. These parameters are now extracted once a model has been trained and the scoring uses only these parameters (e.g. not the libFM executable). 
Don't forget to acknowledge libFM if you publish results produced with this software. Also take note of its licence GNU. More information can be found on [libFM's repo on github](https://github.com/srendle/libfm).

```
OriginalLibFMRegressor type:als lfeatures:4 learn_rate:0.01 maxim_Iteration:10 init_values:0.01 c:1 c2:1 threads:1 usescale:false seed:1 verbose:true bags:1 
```

Parameter | Explanation
--- | ---
Type | Type of algorithm to use. It has to be  sgd, als, mcmc. Default is mcmc.
C | Regularization value, the more, the stronger the regularization. **This is important**
C2 | Regularization value for the latent features. **This is important**
lfeatures | Number of latent features to use. **This is important**
init_values | Initialise values of the latent features with values between[0,init_values). **This is important**
maxim_Iteration | aximum number of iterations. **This is important**
learn_rate | learn_rate for SGD; default=0.1. **This is important**

### VowpaLWabbitRegressor

Wrapper for [vowpal wabbit](https://github.com/JohnLangford/vowpal_wabbit/wiki). It does not contain all features, but a fraction.

```
VowpaLWabbitRegressor use_ftrl:True learning_rate:0.8 decay_learning_rate:0.97 nn:40 bit_precision:18 ftrl_alpha:0.1 ftrl_beta:0.1 power_t:0.9 initial_t:0.9 l1:0.01 l2:0.01 passes:10 threads:1 make2way:false make3way:false use_dropout:true use_meanfield:false seed:1 verbose:true bags:1  
```

Parameter | Explanation
--- | ---
passes | Number of training Passes. **This is important**
bit_precision | number of bits in the feature table. 
decay_learning_rate | Decay factor for learning_rate between passes. 
nn | Number of hidden units to use in a sigmoidal feedforward network with nn hidden units
initial_t | Initial t value. Affects learning rate's updates
power_t | t power value. Affects learning rate's updates
ftrl_alpha | ftrl alpha parameter when using ftrl **This is important**
ftrl_beta | ftrl beta stability patameter when using ftrl **This is important**
learning_rate | learning Rate for gradient-based updates
l1 | L1 regularization 
l2 | L2 regularization **This is important**
use_ftrl | o use the ftrl optimization option (instead of adaptive). It is on by default.
make2way | if true it creates all possible 2-way interactions of all features
make3way | if true it creates all possible 3-way interactions of all features
use_dropout | when nn>0, train or test sigmoidal feedforward network using dropout.
use_meanfield | when nn>0, train or test sigmoidal feedforward network using mean field.

