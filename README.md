# StackNet

This repository contains StackNet Meta modelling methodology (and software) which is part of my work as a PhD Student in the computer science department at [UCL](http://www.cs.ucl.ac.uk/home/)

## What is StackNet

StackNet is a computational, scalable and analytical framework implemented with a software implementation in Java that resembles a feedforward neural network and uses Wolperts stacked generalization [1] in multiple levels to improve accuracy in classification problems. In contrast to feedforward neural networks, rather than being trained through back propagation, the network is built iteratively one layer at a time (using stacked generalization), each of which uses the final target as its target.

The Sofware is made available under MIT licence.

[1] Wolpert, D. H. (1992). Stacked generalization. *Neural networks*, 5(2), 241-259.

##How does it work

Given some input data, a neural network normally applies a perceptron along with a transformation function like relu, sigmoid, tanh or others. The equation is often expressed as :

![Alt text](/images/perceptron.png?raw=true "perceptron input to hidden")

The StackNet model assumes that this function can take the form of any supervised machine learning algorithm - or in other words: 

![Alt text](/images/generic_connection.png?raw=true "Generic input to hidden") 

where *s* expresses that machine learning model. 

Logically the outputs of each neuron, can be fed onto next layers. For instance in the second layer the equation will be : 

![Alt text](/images/second_layer_inputs.png?raw=true "second layer example") 

Where a is one of the H<sub>2</sub> algorithms included in the second layer and can be any estimator, classifier or regressor.

The aforementioned formula could be generalised as follows for any layer: 

![Alt text](/images/normal_stacking.png?raw=true "Generic layer example") 

Where a is the h<sub>th</sub> algorithm out of H in the <sub>n</sub>th hidden (models') layer and f<sub>n-1</sub> the previous models raw score prediction in respect to the target variable. 

To create an output prediction score for any number of unique categories of the response variable, all selected algorithms in the last layer need to have outputs dimensionality equal to the number those unique classes In case where there are many such classifiers, the results is the scaled average of all these output predictions and can be written as: 

![Alt text](/images/generic_output.png?raw=true "Hidden-to-output with many classifiers") 

Where C is the number of unique classifiers in the last layer. In case of just 1 classifier in the output layer this would resemble to the softmax activation function of a typical neural network used for classification. 

## The Modes

The stacking element of the StackNet model could be run with 2 different modes.
 
### Normal stacking mode

 The first mode (e.g. the default) is the one already mentioned and assumes that in each layer uses the predictions (or output scores) of the direct previous one similar with a typical feedforward neural network or equivalently: 
 
![Alt text](/images/normal_stacking.png?raw=true "Normal Stacking") 

### Restacking mode 
 
The second mode (also called restacking) assumes that each layer uses previous neurons activations as well as all previous layers neurons (including the input layer). Therefore the previous formula can be re-written as:  

![Alt text](/images/restacking.png?raw=true "Restacking Mode")

Assuming the algorithm is located in layer n>1, to activate each neuron h in that layer, all outputs from all neurons from the previous n-1 (or k) layers need to be accumulated (or stacked .The intuition behind this mode is drive from the fact that the higher level algorithm have extracted information from the input data, but rescanning the input space may yield new information not obvious from the first passes. This is also driven from the forward training methodology discussed below and assumes that convergence needs to happen within one model iteration.

The modes may also be viewed bellow: 

![Alt text](/images/stacknet_modes.png?raw=true "StackNet's Mode")

## K-fold Training

The typical neural networks are most commonly trained with a form of back propagation, however stacked generalization requites a forward training methodology that splits the data into two parts – one of which is used for training and the other for predictions. The reason this split is necessary is to avoid the over fitting that could be a factor of the kind of algorithms being used as inputs as well as the absolute count of them. 

However splitting the data in just 2 parts would mean that in each new layer the second part needs to be further dichotomized increasing the bias of overfitting even more as each algorithm will have to be trained and validated on increasingly less data. To overcome this drawback the algorithm utilises a k-fold cross validation (where k is a hyper parameter) so that all the original training data is scored in different k batches thereby outputting n shape training predictions where n is the size of the samples in the training data. Therefore the training process is consisted of 2 parts: 

1. Split the data k times and run k models to output predictions for each k part and then bring the k parts back together to the original order so that the output predictions can be used in later stages of the model.

2. Rerun the algorithm on the whole training data to be used later on for scoring the external test data. There is no reason to limit the ability of the model to learn using 100% of the training data since the output scoring is already unbiased (given that it is always scored as a holdout set). 

The K-fold train/predict process is illustrated below: 

![Alt text](/images/kfold_training.png?raw=true "Training StackNet with K-fold")


It should be noted that (1) is only applied during training to create unbiased predictions for the second layerss model to fit one. During scoring time (and after model training is complete) only (2) is in effect.

All models must be run sequentially based on the layers, but the order of the models within the layer does not matter. In other words all models of layer one need to be trained in order to proceed to layer two but all models within the layer can be run asynchronously and in parallel to save time. The k-fold may also be viewed as a form of regularization where smaller number of folds (but higher than 1) ensure that the validation data is big enough to demonstrate how well a single model could generalize. On the other hand higher k means that the models come closer to running with 100% of the training and may yield more unexplained information. The best values could be found through cross validation. Another possible way to implement this could be to save all the k models and use the average of their predicting to score the unobserved test data, but this have all the models never trained with 100% of the training data and may be suboptimal. 

## Some Notes about StackNet

StackNet is (commonly) **better than the best single model it contains in each first layer**, however its ability to perform well still relies on a mix of strong and diverse single models in order to get the best out of this Meta modelling methodology.

StackNet (methodology - not the software) was also used to win the  [Truly Native](http://blog.kaggle.com/2015/12/03/dato-winners-interview-1st-place-mad-professors/ ) data modelling competition hosted by the popular data science platform Kaggle in 2015 

Network's example:

![Alt text](/images/mad_prof_winning.png?raw=true "Winning Truly Native competitions Using STackNet Methodology")

StackNet is made available now with a handful of classifiers and regressors. The implementations are based on the original papers and software . However most have some personal tweaks in them. 


## Algorithms contained (in each first release)

-	AdaboostForestRegressor
-	AdaboostRandomForestClassifier
-	DecisionTreeClassifier
-	DecisionTreeRegressor
-	GradientBoostingForestClassifier
-	GradientBoostingForestRegressor
-	RandomForestClassifier
-	RandomForestRegressor
-	Vanilla2hnnregressor
-	Vanilla2hnnclassifier
-	Softmaxnnclassifier
-	Multinnregressor
-	NaiveBayesClassifier
-	LSVR
-	LSVC
-	LogisticRegression
-	LinearRegression
-	LibFmRegressor
-	LibFmClassifier

Not fully developed

-	knnClassifier 
-	knnRegressor 
-	KernelmodelClassifier 
-	KernelmodelRegressor


## Algorithms’s Tunning parameters

Will add later

## Run StackNet 

You can do so directly from the jar file, using Java higher than 1.6 . You need to add java as an environmental variable (e.g add it to PATH).

The basic format is:

Java –jar stacknet.jar [train or predict] [parameter = value]


### Command Line Paramneters

Command | Explanation
--- | ---
sparse  | True if the data to be imported are in sparse format (libsvm) or dense (false) 
has_head   | True if train_file and test_file have headers else false
model | Name of the output model file. 
pred_file | Name of the output prediction file. 
train_file | Name of the training file. 
test_file | Name f tohe test file. 
output_name | Prefix of the models to be printed per iteration. This is to allow the Meta features of each iteration to be printed. Defaults to nothing.
test_target | True if the test file has a target variable in the beginning (left) else false (only predictors in the file).
params | Parameter file where each line is a model. empty lines correspond to the creation of new levels 
verbose | True if we need StackNet to output its progress else false 
Threads | Number of models to run in parallel. This is independent of any extra threads allocated from the selected algorithms. e.g. it is possible to run 4 models in parallel where one is a randomforest that runs on 10 threads (it selected). 
metric | Metric to output in cross validation for each model-neuron. can be logloss, accuracy or auc (for binary only) 
stackdata | True for restacking else false
seed | Integer for randomised procedures 
folds | Number of folds for re-usable kfold

### Parameters' File

In The parameter file, each line is a model. When there is an empty line then any new algorithm is used in the next level.  This is a sample format. 

```
LogisticRegression C:1 Type:Liblinear maxim_Iteration:100 scale:true verbose:false
RandomForestClassifier bootsrap:false estimators:100 threads:5 logit.offset:0.00001 verbose:false cut_off_subsample:1.0 feature_subselection:1.0 gamma:0.00001 max_depth:8 max_features:0.25 max_tree_size:-1 min_leaf:2.0 min_split:5.0 Objective:ENTROPY row_subsample:0.95 seed:1
GradientBoostingForestClassifier estimators:100 threads: offset:0.00001 verbose:false trees:1 rounding:2 shrinkage:0.05 cut_off_subsample:1.0 feature_subselection:0.8 gamma:0.00001 max_depth:8 max_features:1.0 max_tree_size:-1 min_leaf:2.0 min_split:5.0 Objective:RMSE row_subsample:0.9 seed:1
Vanilla2hnnclassifier UseConstant:true usescale:true seed:1 Type:SGD maxim_Iteration:50 C:0.000001 learn_rate:0.009 smooth:0.02 h1:30 h2:20 connection_nonlinearity:Relu init_values:0.02
LSVC Type:Liblinear threads:1 C:1.0 maxim_Iteration:100 seed:1
LibFmClassifier lfeatures:3 init_values:0.035 smooth:0.05 learn_rate:0.1 threads:1 C:0.00001 maxim_Iteration:15 seed:1
NaiveBayesClassifier usescale:true threads:1 Shrinkage:0.1 seed:1 verbose:false

RandomForestClassifier estimators=1000 rounding:3 threads:4 max_depth:6 max_features:0.6 min_leaf:2.0 Objective:ENTROPY gamma:0.000001 row_subsample:1.0 verbose:false copy=false
```

**Tip** : To tune a single model , one may choose an algorithm for the first layer and a dummy one for the second layer. StackNet expects at least 2 algorithms, so with this format the user can visualize the performance of single algorithm inside the K-fold.
For example, if I wanted to tune a Random Forest Classifier, I would put it in the first line (layer) and also put any model (lets say Logistic Regression) in the second layer and could break the process immediately after the first layer kfold is done:

```
RandomForestClassifier bootsrap:false estimators:100 threads:5 logit.offset:0.00001 verbose:false cut_off_subsample:1.0 feature_subselection:1.0 gamma:0.00001 max_depth:8 max_features:0.25 max_tree_size:-1 min_leaf:2.0 min_split:5.0 Objective:ENTROPY row_subsample:0.95 seed:1

LogisticRegression verbose:false
```

## Data Format

For **dense** input data, the file needs to start with the target variable followed by comma, separated variables like:

1,0,0,2,3,2.4

0,1,1,0,0,12

For **sparse** format , it is the same as libsvm (same example as above)  :

1 2:2 3:3 4:2.4

0 0:1 1:1 4:12

**warning** : Some algorithms (mostly tree-based) may not be very fast with this format)

If test_target is false then the test data may not have a target and start directly from the variables.

A **train** method needs at least a **train_file** and a **params_file**. It also needs at least 2 algorithms and the and last layer must not contain a regressor unless the metric is auc and the problem is binary.

A **predict** method needs at least a **test_file** and a **model_file**. 

### Training example:

Java –jar stacknet.jar **_train_** **sparse**=false **has_head**=true **model**=model **pred_file**=pred.csv **train_file**=sample_train.csv **test_file**= sample_test.csv **test_target**=true **params**=params.txt **verbose**=true **threads**=7 **metric**=logloss **stackdata**=false **seed**=1 **folds**=5

Note that you can have train and test at the same time. In that case after training, it scores the test data. 

### Predict example:

Java –jar stacknet.jar **_predict_** **sparse**=false **has_head**=true **model**=model **pred_file**=pred.csv **test_file**=sample_test.csv **test_target**=true **verbose**=true  **metric**=logloss


## Run StackNet from within (Java) code


If we wanted to build a 3-level stacknet on a binary target with desne data, we start with initializing a _StackNetClassifier_ Object:

```java
 StackNetClassifier StackNet = new StackNetClassifier (); // Initialise a StackNet 
``` 

Which is then followed by the a 2 dimensional String array with the list of models in each layer along with their hyper parameters in the form of as in "_estimator [space delimited hyper parameters]_"

```java	
String models_per_level[][]=new String[][]; 

			
{//First Level
{"LogisticRegression C:0.5 maxim_Iteration:100 verbose:true", 
"RandomForestClassifier bootsrap:false estimators:100 threads:25 offset:0.00001 cut_off_subsample:1.0 feature_subselection:1.0 max_depth:15 max_features:0.3 max_tree_size:-1 min_leaf:2.0 min_split:5.0 Objective:ENTROPY row_subsample:0.95", 
"LSVC C:3 maxim_Iteration:50",
"LibFmClassifier maxim_Iteration:16 C:0.000001 lfeatures:3 init_values:0.9 learn_rate:0.9 smooth:0.1", 
"NaiveBayesClassifier Shrinkage:0.01", 
"Vanilla2hnnclassifier maxim_Iteration:20 C:0.000001 tolerance:0.01 learn_rate:0.009 smooth:0.02 h1:30 h2:20 connection_nonlinearity:Relu init_values:0.02", 
"GradientBoostingForestClassifier estimators:100 threads:25 verbose:false trees:1 rounding:2 shrinkage:0.1 feature_subselection:0.5 max_depth:8 max_features:1.0 min_leaf:2.0 min_split:5.0 row_subsample:0.9", 
"LinearRegression C:0.00001", 
"AdaboostRandomForestClassifier estimators:100 threads:3 verbose:true trees:1 rounding:2 weight_thresold:0.4 feature_subselection:0.5 max_depth:8 max_features:1.0 min_leaf:2.0 min_split:5.0 row_subsample:0.9", 
"GradientBoostingForestRegressor estimators:100 threads:3 trees:1 rounding:2 shrinkage:0.1 feature_subselection:0.5 max_depth:9 max_features:1.0 min_leaf:2.0 min_split:5.0 row_subsample:0.9", 
"RandomForestRegressor estimators:100 internal_threads:1 threads:25 offset:0.00001 verbose:true cut_off_subsample:1.0 feature_subselection:1.0 max_depth:14 max_features:0.25 max_tree_size:-1 min_leaf:2.0 min_split:5.0 Objective:RMSE row_subsample:1.0", 
"LSVR C:3 maxim_Iteration:50 P:0.2" },
//Second Level				
{"RandomForestClassifier estimators:1000  threads:25 offset:0.0000000001 verbose=false cut_off_subsample:0.1 feature_subselection:1.0 max_depth:7 max_features:0.4  max_tree_size:-1 min_leaf:1.0  min_split:2.0 Objective:ENTROPY row_subsample:1.0",
"GradientBoostingForestClassifier estimators:1000 threads:25 verbose:false trees:1 rounding:4 shrinkage:0.01 feature_subselection:0.5 max_depth:5 max_features:1.0 min_leaf:1.0 min_split:2.0 row_subsample:0.9",	
"Vanilla2hnnclassifier maxim_Iteration:20 C:0.000001 tolerance:0.01 learn_rate:0.009 smooth:0.02 h1:30 h2:20 connection_nonlinearity:Relu init_values:0.02",	
"LogisticRegression C:0.5 maxim_Iteration:100 verbose:false" },
//Third Level					
{"RandomForestClassifier estimators:1000  threads:25 offset:0.0000000001 verbose=false cut_off_subsample:0.1 feature_subselection:1.0 max_depth:6 max_features:0.7  max_tree_size:-1 min_leaf:1.0  min_split:2.0 Objective:ENTROPY row_subsample:1.0" }
};

Alternatively, we could load directly from a file :

String modellings[][]=io.input.StackNet_Configuration("params.txt");

StackNet.parameters=models_per_level; // adding the models' specifications
```

The remaining parameters to be specified include the cross validation training schema, the Restacking mode option, setting a random state as well as some other miscellaneous options: 

```java		
StackNet.threads=4; // models to be run in parallel
StackNet.folds=5; // size of K-Fold
StackNet.stackdata=true; // use Restacking
StackNet.print=true; // this helps to avoid rerunning should the model fail
StackNet.output_name="restack";// prefix for each layer's output.
StackNet.verbose=true; // it outputs 
StackNet.seed=1; // random state
StackNet.metric="logloss"

```

Ultimately given a data object X and a 1 dimensional vector y, the model can be trained using :

```java
StackNet.target=y; // the target variable		
StackNet.fit(X); // fitting the model on the training data

```
Predictions are made with :

```java
double preds [][]=StackNet.predict_proba(X_test);
```

## Potential Next Steps

- Add StackNetRegressor
- increase coverage with well-known and well-performing ml tools
- add preprocessing steps, feature engineering , feature selections

## Reference

For now you may use this :

Marios Michailidis (2017), StackNet, StackNet Meta Modelling Framework, url https://github.com/kaz-Anova/StackNet
