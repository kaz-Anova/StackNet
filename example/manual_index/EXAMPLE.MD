# Example that uses manual index

This example explores the newest features of StackNet (version 1.04). Namely it looks at `input_index`, `inlcude_target` and includes models from libfm, libffm and vowpal wabbit. 
The `input_index` expects a file (index_file.txt) that specifies the cv fold of each row. This file needs to have the same row size as the training file. It excludes headers. 
The `inlcude_target=true` command just appends the target variable in the holdout prediction files if the `output_name` command is used.

To run follow the next steps:

1. Ensure you have **train.csv**, **test.csv**, *params.txt** and **index_file.txt** in the same folder with **StackNet.jar** and the **lib/** folder. The .csv files have dense format, they include headers and the target variable in the beginning  
2. You need to follow the instructions to make certain that the executables for the algorithms are working. Specifically, you need to check [libfm](https://github.com/kaz-Anova/StackNet#Install-original-libfm), [vowpal wabbit](https://github.com/kaz-Anova/StackNet#Install-vowpal-wabbit), [libffm](https://github.com/kaz-Anova/StackNet#Install-libffm). **Please don't skip this step**. Most errors are due to the fact that the binaries are not working properly.    
3. Make certain you have **Java** higher than 1.6 installed and that Java is in your **PATH**. Have a look at [this] (https://www.java.com/en/download/help/path.xml) if you encounter trouble.
4. Run StackNet with `java -jar StackNet.jar train task=regression train_file=train.csv test_file=test.csv params=params.txt pred_file=pred.csv output_name=models input_index=index_file.txt bin=2 include_target=true test_target=true has_head=true verbose=true Threads=1 folds=4 metric=rmse` . Note there `folds=4`, however in **index_file.txt** there are 5 folds. 5 Folds will be used, the `folds=4` is ignored. 
5. The holdout predictions (in **models1.csv**) inlcude an extra column with the target in the beginning. 7 columns in total. 
