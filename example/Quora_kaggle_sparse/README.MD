# Example with Sparse data based on Quora Question Pairs

This example uses [StackNet](https://github.com/kaz-Anova/StackNet) along with some preparation with python to score 0.3250 (logloss or better after merging with another [kernel submission](https://www.kaggle.com/woters/quora-question-pairs/xgb-starter-12357/output) ) for the [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) challenge on [kaggle](https://www.kaggle.com/) 

To run follow the next steps:

1. Download the **train.csv** and **test.csv** files from [here](https://www.kaggle.com/c/quora-question-pairs/data)
2. Download the **StackNet.jar** file from the Git
3. Ensure you have **Python** installed 
4. Make certain you have **Java** higher than 1.6 installed and that Java is in your **PATH**. Have a look at [this](https://www.java.com/en/download/help/path.xml) if you encounter trouble.
5. Run the **main_querry_v1.py** script (also in the example section of the git). This is based on this [script]( https://www.kaggle.com/thedarkl0rd/quora-question-pairs/xgbstarter-340/code) . It creates the same basic data as in the script plus a simple **tf-idf** on question1 and question2 merged as the main purpose was to test high-cardinality sparse data. Then it prints the **train.sparse** and **test.sparse** in Libsvm format in the same folder where the file is executed. 
6. **Run in the command line**. You may need to look at the [parameters’](https://github.com/kaz-Anova/StackNet) section on GitHub to understand more about the available models and their hyper paramaters. CD to the folder where all the files are and press: 
*java -Xmx4048m -jar StackNet.jar train task=classification train_file=train.sparse test_file=test.sparse params=paramsv1.txt sparse=True pred_file=querry_pred.csv test_target=false verbose=true Threads=3 stackdata=false folds=5 seed=1 metric=logloss*
7. The whole process with **3 threads** may take **around 4-5 hours**. Consider increasing the **threads** to **11** if you have that many, but you might need to increase the allocated memory from 4 GB (**Xmx4048m**) to 6 GB (**Xmx6072m**) or more . 
8. The structure as appears in the **paramsv1.txt** file contains 11 level_1 models and 1 level_2 model. You can learn more about the available models and parameters [here]( https://github.com/kaz-Anova/StackNet/blob/master/parameters/PARAMETERS.MD)  
9. You can get **0.31019** via averaging the output prediction file of StackNet, in this scenario **querry_pred.csv** with the ** xgb_seed12357_n315.csv** from [this kernel](https://www.kaggle.com/woters/quora-question-pairs/xgb-starter-12357/output) . You may use the **create_sub_from_preds.py** to do that. If you want only the StackNet prediction, then set the **second_file** to “”
10. There is an ongoing discussion with many tips and troubleshooting in the [twosigma](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/30012) competition too. 
