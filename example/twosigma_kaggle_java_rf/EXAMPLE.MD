# Example based on twosigma competition

This example use **RandomForestClassifier** directly from Java
This example assumes you are familiar with adding external Jars to your Java projects and generally you have experience with Java

The first 6 steps are the same as in the [StackNet twosigma example](https://github.com/kaz-Anova/StackNet/blob/master/example/twosigma_kaggle/EXAMPLE.MD)

To run follow the next steps:

1. Download the **train.json** and **test.json** files from [here] (https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data)
2. Download the **StackNet.jar** file from the Git
3. Ensure you have **Python** installed along with a fairly recent version of **xgboost**
4. Make certain you have **Java** higher than 1.6 installed and that Java is in your **PATH**. Have a look at [this] (https://www.java.com/en/download/help/path.xml) if you encounter trouble.
5. Run the **create_files_v1.py**  script (also in the example section of the git). This is based on [SRK’s starter python script](https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/xgb-starter-in-python) . It creates some data from the raw json files, runs an xgboost (which scores around 0.544 to LB)and stacks xgboost’s predictions onto the data and then prints them out as csvs (e.g. train_stacknet.csv and test_stacknet.csv) . The intuition is mostly speed as well as the fact that xgboost is too good and since it is not part of the initial models on StackNet – I had to somehow add it!
6. Put the jar file, the attached parameters’ file (paramssimplev1.txt) , the printed csv files into the **same folder**. The parameter files contains 9 level 1 models and 1 level 2 model. All models within the level are built in parallel. 

The rest in new:

7. If you use an IDE (like eclispe) , create a project and then add the **Stackent_source.jar** file from the Git
8. Ensure your Java compiler level is 1.6 or more
9. Import the **rf_stacknet.java**
10. Make changes to the path/directry names if you have to
11. the output file named as rf_Submission_0. etc scores x on LB . It scores **0.53789**

It should print something like this :

```java		
 cv fold: 1/5  training size: 39482 test size: 9870 logloss-----> ( 0.5329) 
 cv fold: 2/5  training size: 39482 test size: 9870 logloss-----> ( 0.5338) 
 cv fold: 3/5  training size: 39482 test size: 9870 logloss-----> ( 0.5296) 
 cv fold: 4/5  training size: 39482 test size: 9870 logloss-----> ( 0.5317) 
 cv fold: 5/5  training size: 39480 test size: 9872 logloss-----> ( 0.5340) 
 Final logloss-----> 0.5324 <-----

```
