# Example based on kaggle's Amazon.com - Employee Access Challenge 

This example uses 2 StackNet models to achieve a top 11 score withn 2 hours (on 1 thread) in the popular kaggle challenge.

![Alt text](/example/example_amazon/images/top_11_score.png?raw=true "top 11 score") 

* Amazon.com - Employee Access Challenge was a popular Kaggle competition (around 1700 teams).
* This code will get you around top10 within a few hours (including data preparation and modelling) via using 2 StackNet models.
* The interesting about this competition is it has only 8 Variables (and 1 duplicate)! 
* All categorical with high cardinality
* The purpose is to build a model, learned using historical data, that will determine an employee's access needs 
* First StackNet model is based on a dataset that finds the top 4way interractions of all the features, which then is binarized using onehot encoding. Therfore the input data is **sparse**.
* The second model makes use of the new **data_prefix** command which tells StackNet to expect different pairs of train and validation data to run stacking on. In other words the User supplies the data per fold. This schema is used because likelihood features and counts are created within cross-validation.
* The metric to optimize is Area Under The Roc Curve or simply AUC

To get the top the top score , we start with the StackNet model based on the 5way interractions (sparse)

1. Data is downloaded from [here](https://www.kaggle.com/c/amazon-employee-access-challenge) 
2. Run with python the **prepare_data.py** script to create the modelling data. 
3. Assuming you have the **StackNet.jar** in the same folder with these files and **param_amazon_linear.txt** is also in the same folder, run this in the command line : *java -Xmx3048m -jar StackNet.jar train task=classification train_file=train.sparse test_file=test.sparse params=param_amazon_linear.txt pred_file=amazon_linear_pred.csv test_target=false verbose=true Threads=1 sparse=true folds=5 seed=1 metric=auc* 
4. The performance of the models is as follows:

model | auc
--- | ---
**LogisticRegression_L2**  | **0.893**
LogisticRegression_L2_SGD  | 0.885
LSVC_L2  | 0.891
LinearRegression  | 0.879
LibFmClassifier  | 0.891
softmaxnnclassifier  | 0.882
GradientBoostingForestClassifier  | 0.851
LogisticRegression_L1  | 0.88
LSVC_L1   | 0.871

RoC of best Logistic model:

![Alt text](/example/example_amazon/images/best_linear_model.png?raw=true "best linear model") 

The RandomForest Meta Classifier in level 2 had **AUC of 0.901** which is about **0.008 gain** from the top performing model.

The second part will use the data per fold 
5. Execute from the command line : *java -Xmx3048m -jar StackNet.jar train task=classification data_prefix=amazon_counts test_file=amazon_counts_test.txt  pred_file=amazon_count_pred.csv  verbose=true Threads=1 folds=5 seed=1 metric=auc*


model | auc
--- | ---
LogisticRegression  | 0.889
**GradientBoostingForestClassifier**  | **0.9**
RandomForestClassifier  | 0.899
softmaxnnclassifier  | 0.866
LSVC  | 0.888
LibFmClassifier  | 0.89
GradientBoostingForestRegressor  | 0.858
LinearRegression  | 0.901

RoC of best (GBM)  model:

![Alt text](/example/example_amazon/images/best_count_model.png?raw=true "best count model") 

The RandomForest Meta Classifier in level 2 had **AUC of 0.904** which is about **0.004 gain** from the top performing model (less than the linear blend).

6. run the **blend_script.py** to blend the results of the two StackNets
7. submit
8. Try your own things - add your own models! There is room for improvement :)  

There is also another example done in the past with python for the same data that displays and compares different ensemble techniques, including stacking [here](https://github.com/kaz-Anova/ensemble_amazon)

