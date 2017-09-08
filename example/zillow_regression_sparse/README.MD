# Example with regression and Sparse data based on Zillow

This example uses [StackNet](https://github.com/kaz-Anova/StackNet) along with some preparation with python to score 0.0647+ (mae or better after merging with another [kernel submission](https://www.kaggle.com/davidfumo/boosted-trees-lb-0-0643707/code) for the [Zillow Prize: Zillow’s Home Value Prediction](https://www.kaggle.com/c/zillow-prize-1/) challenge on [kaggle](https://www.kaggle.com/) 

(*Some of the models do not accept seeds and final score may vary a bit.*)

To run follow the next steps:

1. Download the all files from [here](https://www.kaggle.com/c/zillow-prize-1/data) 
2. Clone the git as `git clone https://github.com/kaz-Anova/StackNet.git`
3. Ensure you have **Python** installed and that Python can be found on **PATH** as StackNet now makes python calls as subprocesses. Also ensure that **sklearn 0.18** or above is installed. If you have trouble doing it in windows, see [this](https://www.youtube.com/watch?v=Y2q_b4ugPWk).  
4. Make certain you have **Java** higher than 1.6 installed and that Java is in your **PATH**. Have a look at [this](https://www.java.com/en/download/help/path.xml) if you encounter trouble.
5. **Either** create a new (base) directory where you will put in the **StackNet.jar** file AND the **lib/** folder **OR** just work inside the cloned github repo. Whichever one you chose, now create an `input` folder where you put  all data from (1). Additionally copy and all files contained in this example (**not inside** the input folder, just in the base directory where you will execute the code from). In principle the only thing you need to run StackNet is the ** StackNet.jar** file and the **lib/** folder. The **lib/** folder contains the binaries to run some of the newer wrappers (xgboost, lightgbm, fastrgf).
6. From the base directory execute the **make_stacknet_data.py** as `python  make_stacknet_data.py` to generate 2 files in sparse format (*dataset2_train.txt,* *dataset2_test.txt*) . This is taken from [here](https://www.kaggle.com/danieleewww/xgboost-lightgbm-and-olsv107-w-month-features). It is the middle dataset in that code (the one that applies outlier removal). 
6. Optionally, it will be good to do some sanity checks to ensure that all binaries work fine for [xgboost](https://github.com/kaz-Anova/StackNet#install-xgboost), [lightgbm](https://github.com/kaz-Anova/StackNet#install-lightgbm), [fast_rgf](https://github.com/kaz-Anova/StackNet#install-fast_rgf-new) and [python algos](https://github.com/kaz-Anova/StackNet#install-sklearn-algorithmsnew) . There is high chance linux users will need to `chmod +x`  the executables inside the lib/[your_os]/[tools_name]/
7. **Run in the command line**. You may need to look at the [parameters’](https://github.com/kaz-Anova/StackNet#command-line-parameters) section on GitHub to understand more about the available models and their hyper paramaters. From the base directory run 
*java -Xmx12048m -jar StackNet.jar train task=regression sparse=true has_head=false output_name=datasettwo model=model2 pred_file=pred2.csv train_file=dataset2_train.txt test_file=dataset2_test.txt test_target=false params=dataset2_params.txt verbose=true threads=1 metric=mae stackdata=false seed=1 folds=4 bins=3*
8. The whole process may take **around 2 hours**. Consider increasing the **threads** in the command or from inside the **dataset2_params.txt** (which defines the network’s structure). There you can increase the threads on individual algorithms – this method is preferred as it less expensive (e.g one model finishes before the other one starts).  You  might need to increase the allocated memory to 16 GB (**Xmx16048m**) or more.**FRGFRegressor** uses lots of memory, so consider removing it if it crashes due to memory.   
9. The structure as appears in the **dataset2_params .txt** file contains 12 level_1 models and 1 level_2 model. You can learn more about the available models and parameters [here]( https://github.com/kaz-Anova/StackNet/blob/master/parameters/PARAMETERS.MD)

The architecture of this ensemble is illustrated bellow.

![Alt text](/example/zillow_regression_sparse/images/structure.png?raw=true "architecture")


10. your predictions are now saved in **pred2.csv** in the base directory.
11. You can generate a submission via executing the **create_submission.py** as `python create_submission.py`. It will be saved to *output_dataset2.csv*
12. From now on, everything is optional. You can get a score of 0.0643+ (top 3% at the time) if you blend with the output of [another script](https://www.kaggle.com/davidfumo/boosted-trees-lb-0-0643707/output).
13. Download **sub20170821_221910.csv** from the link in (12) and place it into the base directory.
14. You can generate a submission combining the 2 submissions via executing the **combine_subs.py** as `python combine_subs.py`. It will be saved to *output_dataset2_merged.csv*
15. Again optionally you can get the great genetic programming script from [here](https://www.kaggle.com/scirpus/genetic-programming-lb-0-0643904/output) and unzip it to get to 0.06425+(top 40 the time of post) after merging with the previous submission. The file tha comes out after unzipping is teh **xxx.csv**.
16.  You can generate a submission combining the 2 submissions via executing the **combine_subs_v2.py** as `python combine_subs_v2.py`. It will be saved to *output_dataset2_merged_v2.csv*

This is how your base directory needs to look like after everything is done: 

![Alt text](/example/zillow_regression_sparse/images/tree.png?raw=true "architecture")

