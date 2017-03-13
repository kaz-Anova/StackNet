# Example based on twosigma competition

This example use StackNet along with some preparation with python along with [xgboost](https://github.com/dmlc/xgboost) to score 0.53776 (logloss) for the [twosigma] (https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries)) challenge on [kaggle](https://www.kaggle.com/) 

To run follow the next steps:

1. Download the **train.json** and **test.json** files from [here] (https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data)
2. Download the **StackNet.jar** file from the Git
3. Ensure you have **Python** installed along with a fairly recent version of **xgboost**
4. Make certain you have **Java** higher than 1.6 installed and that Java is in your **PATH**. Have a look at [this] (https://www.java.com/en/download/help/path.xml) if you encounter trouble.
5. Run the **create_files_v1.py**  script (also in the example section of the git). This is based on [SRK’s starter python script](https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/xgb-starter-in-python) . It creates some data from the raw json files, runs an xgboost (which scores around 0.544 to LB)and stacks xgboost’s predictions onto the data and then prints them out as csvs (e.g. train_stacknet.csv and test_stacknet.csv) . The intuition is mostly speed as well as the fact that xgboost is too good and since it is not part of the initial models on StackNet – I had to somehow add it!
6. Put the jar file, the attached parameters’ file (paramssimplev1.txt) , the printed csv files into the **same folder**. The parameter files contains 9 level 1 models and 1 level 2 model. All models within the level are built in parallel. 
7. **Run on the command line**. You may need to look on the parameters’ section on GitHub to understand what is everything. CD to that folder while in command line and press: 
   *java -Xmx3048m -jar StackNet.jar train train_file=train_stacknet.csv test_file=test_stacknet.csv params=paramssimplev1.txt pred_file=sigma_stack_pred.csv test_target=true verbose=true Threads=3 stackdata=false folds=5 seed=1 metric=logloss*
8. The test_stacknet.csv’s first column is the id. **Append** it next to the predicted probabilities as they come out from StackNet (sigma_stack_pred.csv) and submit . It should score **0.53776** . 
9. You can  get **0.52879** if you average it 50-50 with [this script](https://www.kaggle.com/menghanli0418/two-sigma-connect-rental-listing-inquiries/it-is-lit/output) , but bear in mind that the ids are sorted differently in the two files. So you might need to sort them before joining. 
