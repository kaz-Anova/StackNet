# StackNet

This repository contains StackNet Meta modelling methodology (and software) which is part of my work as a PhD Student in the computer science department at [UCL](http://www.cs.ucl.ac.uk/home/).  My PhD was sponsored by [dunnhumby](http://www.dunnhumby.com/).

StackNet is empowered by [H2O](https://github.com/h2oai/h2o-3)'s [agorithms](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html)

StackNet and other topics can now  be discussed on [FaceBook](https://www.facebook.com/StackNet/) too :

## Contents

-    [What is StackNet](#what-is-stacknet)
-    [How does it work](#how-does-it-work)
-    [The Modes](#the-modes)
-    [Some Notes about StackNet](#some-notes-about-stacknet)
-    [Algorithms contained](#algorithms-contained)
-    [Algorithm's Tuning parameters](#algorithms-tuning-parameters)
-    [Run StackNet](#run-stacknet)
-    [Installations](#installations)
-    [Command Line Parameters](#command-line-parameters)
-    [Data Format](#data-format)
-    [Commandline Train Statement](#commandline-train-statement)
-    [Commandline predict Statement](#commandline-predict-statement)
-    [Examples](#examples)
-    [Run StackNet from within Java code](#run-stacknet-from-within-java-code)
-    [Potential Next Steps](#potential-next-steps)
-    [Reference](#reference)
-    [News](#news)
-    [Special Thanks](#special-thanks)


![Alt text](/images/StackNet_Logo.png?raw=true "StackNet Logo")

## What is StackNet

StackNet is a computational, scalable and analytical framework implemented with a software implementation in Java that resembles a feedforward neural network and uses Wolpert's stacked generalization [1] in multiple levels to improve accuracy in machine learning problems. In contrast to feedforward neural networks, rather than being trained through back propagation, the network is built iteratively one layer at a time (using stacked generalization), each of which uses the final target as its target.

The Sofware is made available under MIT licence.

[1] Wolpert, D. H. (1992). Stacked generalization. *Neural networks*, 5(2), 241-259.

## How does it work

Given some input data, a neural network normally applies a perceptron along with a transformation function like relu, sigmoid, tanh or others. The equation is often expressed as:

![Alt text](/images/perceptron.png?raw=true "perceptron input to hidden")

The StackNet model assumes that this function can take the form of any supervised machine learning algorithm - or in other words: 

![Alt text](/images/generic_connection.png?raw=true "Generic input to hidden") 

where *s* expresses that machine learning model and *g* (normally) a linear activation function or identity

Logically the outputs of each neuron, can be fed onto next layers. For instance in the second layer the equation will be : 

![Alt text](/images/second_layer_inputs.png?raw=true "second layer example") 

Where a is one of the H<sub>2</sub> algorithms included in the second layer and can be any estimator, classifier or regressor.

The aforementioned formula could be generalised as follows for any layer: 

![Alt text](/images/normal_stacking.png?raw=true "Generic layer example") 

Where a is the h<sub>th</sub> algorithm out of H in the <sub>n</sub>th hidden (models') layer and f<sub>n-1</sub> the previous models raw score prediction in respect to the target variable. 

To create an output prediction score for any number of unique categories of the response variable, all selected algorithms in the last layer need to have outputs dimensionality equal to the number those unique classes In case where there are many such classifiers, the results is the scaled average of all these output predictions and can be written as: 

![Alt text](/images/generic_output.png?raw=true "Hidden-to-output with many classifiers") 

Where C is the number of unique classifiers in the last layer. In the case of just one classifier in the output layer, this would resemble the softmax activation function of a typical neural network used for classification. 

## The Modes

The stacking element of the StackNet model could be run with two different modes.
 
### Normal stacking mode

 The first mode (e.g. the default) is the one already mentioned and assumes that in each layer uses the predictions (or output scores) of the direct previous one similar with a typical feedforward neural network or equivalently: 
 
![Alt text](/images/normal_stacking.png?raw=true "Normal Stacking") 

### Restacking mode 
 
The second mode (also called restacking) assumes that each layer uses previous neurons activations as well as all previous layers neurons (including the input layer). Therefore the previous formula can be re-written as:  

![Alt text](/images/restacking.png?raw=true "Restacking Mode")

Assuming the algorithm is located in layer n>1, to activate each neuron h in that layer, all outputs from all neurons from the previous n-1 (or k) layers need to be accumulated (or stacked). The intuition behind this mode is derived from the fact that the higher level algorithm has extracted information from the input data, but rescanning the input space may yield new information not obvious from the first passes. This is also driven from the forward training methodology discussed below and assumes that convergence needs to happen within one model iteration.

The modes may also be viewed bellow: 

![Alt text](/images/stacknet_modes.png?raw=true "StackNet's Mode")

## K-fold Training

The typical neural networks are most commonly trained with a form of backpropagation, however, stacked generalization requires a forward training methodology that splits the data into two parts – one of which is used for training and the other for predictions. The reason this split is necessary is to avoid the overfitting that could be a factor of the kind of algorithms being used as inputs as well as the absolute count of them. 

However splitting the data into just two parts would mean that in each new layer the second part needs to be further dichotomized increasing the bias of overfitting even more as each algorithm will have to be trained and validated on increasingly fewer data. To overcome this drawback, the algorithm utilises a k-fold cross validation (where k is a hyperparameter) so that all the original training data is scored in different k batches thereby outputting n shape training predictions where n is the size of the samples in the training data. Therefore the training process consists of two parts: 

1. Split the data k times and run k models to output predictions for each k part and then bring the k parts back together to the original order so that the output predictions can be used in later stages of the model.

2. Rerun the algorithm on the whole training data to be used later on for scoring the external test data. There is no reason to limit the ability of the model to learn using 100% of the training data since the output scoring is already unbiased (given that it is always scored as a holdout set). 

The K-fold train/predict process is illustrated below: 

![Alt text](/images/kfold_training.png?raw=true "Training StackNet with K-fold")


It should be noted that (1) is only applied during training to create unbiased predictions for the second layers model to fit one. During the scoring time (and after model training is complete) only (2) is in effect.

All models must be run sequentially based on the layers, but the order of the models within the layer does not matter. In other words, all models of layer one need to be trained to proceed to layer two but all models within the layer can be run asynchronously and in parallel to save time. The k-fold may also be viewed as a form of regularization where a smaller number of folds (but higher than 1) ensure that the validation data is big enough to demonstrate how well a single model could generalize. On the other hand higher k means that the models come closer to running with 100% of the training and may yield more unexplained information. The best values could be found through cross-validation. Another possible way to implement this could be to save all the k models and use the average of their predicting to score the unobserved test data, but this has all the models never trained with 100% of the training data and may be suboptimal. 

## Some Notes about StackNet

StackNet is (commonly) **better than the best single model it contains in each first layer** however, its ability to perform well still relies on a mix of strong and diverse single models in order to get the best out of this Meta modelling methodology.

StackNet (methodology - not the software) was also used to win the  [Truly Native](http://blog.kaggle.com/2015/12/03/dato-winners-interview-1st-place-mad-professors/ ) data modelling competition hosted by the popular data science platform Kaggle in 2015 

StackNet in simple terms is also explained in [kaggle's blog](http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/) 

Network's example:

![Alt text](/images/mad_prof_winning.png?raw=true "Winning Truly Native competitions Using STackNet Methodology")

StackNet is made available now with a handful of classifiers and regressors. The implementations are based on the original papers and software. However, most have some personal tweaks in them. 


## Algorithms contained


### Native
-    [AdaboostForestRegressor](/parameters/PARAMETERS.MD#adaboostforestregressor)
-    [AdaboostRandomForestClassifier](/parameters/PARAMETERS.MD#adaboostrandomforestclassifier)
-    [DecisionTreeClassifier](/parameters/PARAMETERS.MD#decisiontreeclassifier)
-    [DecisionTreeRegressor](/parameters/PARAMETERS.MD#decisiontreeregressor)
-    [GradientBoostingForestClassifier](/parameters/PARAMETERS.MD#gradientboostingforestclassifier)
-    [GradientBoostingForestRegressor](/parameters/PARAMETERS.MD#gradientboostingforestregressor)
-    [RandomForestClassifier](/parameters/PARAMETERS.MD#randomforestclassifier)
-    [RandomForestRegressor](/parameters/PARAMETERS.MD#randomforestregressor)
-    [Vanilla2hnnregressor](/parameters/PARAMETERS.MD#vanilla2hnnregressor)
-    [Vanilla2hnnclassifier](/parameters/PARAMETERS.MD#vanilla2hnnclassifier)
-    [Softmaxnnclassifier](/parameters/PARAMETERS.MD#softmaxnnclassifier)
-    [Multinnregressor](/parameters/PARAMETERS.MD#multinnregressor)
-    [NaiveBayesClassifier](/parameters/PARAMETERS.MD#naivebayesclassifier)
-    [LSVR](/parameters/PARAMETERS.MD#lsvr)
-    [LSVC](/parameters/PARAMETERS.MD#lsvc)
-    [LogisticRegression](/parameters/PARAMETERS.MD#logisticregression)
-    [LinearRegression](/parameters/PARAMETERS.MD#linearregression)
-    [LibFmRegressor](/parameters/PARAMETERS.MD#libfmregressor)
-    [LibFmClassifier](/parameters/PARAMETERS.MD#libfmclassifier)

### Native - Not fully developed

-    knnClassifier 
-    knnRegressor 
-    KernelmodelClassifier 
-    KernelmodelRegressor

### Wrappers
-    [XgboostRegressor](/parameters/PARAMETERS.MD#xgboostregressor)
-    [XgboostClassifier](/parameters/PARAMETERS.MD#xgboostclassifier)
-    [LightgbmRegressor](/parameters/PARAMETERS.MD#lightgbmregressor)
-    [LightgbmClassifier](/parameters/PARAMETERS.MD#lightgbmclassifier)
-    [FRGFRegressor](/parameters/PARAMETERS.MD#frgfegressor)
-    [FRGFClassifier](/parameters/PARAMETERS.MD#frgfClassifier)
-    [OriginalLibFMClassifier](/parameters/PARAMETERS.MD#originallibfmclassifier)(**New**)
-    [OriginalLibFMRegressor](/parameters/PARAMETERS.MD#originallibfmregressor)(**New**)
-    [VowpaLWabbitClassifier](/parameters/PARAMETERS.MD#vowpalwabbitclassifier)(**New**)
-    [VowpaLWabbitRegressor](/parameters/PARAMETERS.MD#vowpalwabbitregressor)(**New**)
-    [libffmClassifier](/parameters/PARAMETERS.MD#libffmclassifier)(**New**)


### H2O
-  [H2ODeepLearningClassifier](/parameters/PARAMETERS.MD#h2odeeplearningclassifier)
-  [H2ODeepLearningRegressor](/parameters/PARAMETERS.MD#h2odeeplearningregressor)
-  [H2ODrfClassifier](/parameters/PARAMETERS.MD#h2odrfclassifier)
-  [H2ODrfRegressor](/parameters/PARAMETERS.MD#h2odrfregressor)
-  [H2OGbmClassifier](/parameters/PARAMETERS.MD#h2ogbmclassifier)
-  [H2OGbmRegressor](/parameters/PARAMETERS.MD#h2ogbmregressor)
-  [H2OGlmClassifier](/parameters/PARAMETERS.MD#h2oglmclassifier)
-  [H2OGlmRegressor](/parameters/PARAMETERS.MD#h2oglmregressor)
-  [H2ONaiveBayesClassifier](/parameters/PARAMETERS.MD#h2onaivebayesclassifier)

### Python

#### Sklearn(**New**)
-  [SklearnAdaBoostClassifier](/parameters/PARAMETERS.MD#sklearnadaboostclassifier)
-  [SklearnAdaBoostRegressor](/parameters/PARAMETERS.MD#sklearnadaboostregressor)
-  [SklearnDecisionTreeClassifier](/parameters/PARAMETERS.MD#sklearndecisiontreeclassifier)
-  [SklearnDecisionTreeRegressor](/parameters/PARAMETERS.MD#sklearndecisiontreeregressor)
-  [SklearnExtraTreesClassifier](/parameters/PARAMETERS.MD#sklearnextratreesclassifier)
-  [SklearnExtraTreesRegressor](/parameters/PARAMETERS.MD#sklearnextratreesregressor)
-  [SklearnknnClassifier](/parameters/PARAMETERS.MD#sklearnknnclassifier)
-  [SklearnknnRegressor](/parameters/PARAMETERS.MD#sklearnknnregressor)
-  [SklearnMLPClassifier](/parameters/PARAMETERS.MD#sklearnmlpclassifier)
-  [SklearnMLPRegressor](/parameters/PARAMETERS.MD#sklearnmlpregressor)
-  [SklearnRandomForestClassifier](/parameters/PARAMETERS.MD#sklearnrandomforestclassifier)
-  [SklearnRandomForestRegressor](/parameters/PARAMETERS.MD#sklearnrandomforestregressor)
-  [SklearnSGDClassifier](/parameters/PARAMETERS.MD#sklearnsgdclassifier)
-  [SklearnSGDRegressor](/parameters/PARAMETERS.MD#sklearnsgdregressor)
-  [SklearnsvmClassifier](/parameters/PARAMETERS.MD#sklearnsvmclassifier)
-  [SklearnsvmRegressor](/parameters/PARAMETERS.MD#sklearnsvmregressor)


#### Keras
-  [KerasnnClassifier](/parameters/PARAMETERS.MD#kerasnnclassifier)
-  [KerasnnRegressor](/parameters/PARAMETERS.MD#kerasnnregressor)

#### Generic for user defined scripts (**New**)
-  [PythonGenericClassifier](/parameters/PARAMETERS.MD#pythongenericclassifier)
-  [PythonGenericRegressor](/parameters/PARAMETERS.MD#pythongenericregressor)

## Algorithm's Tuning parameters

For the common models, have a look at:

[parameters](/parameters/PARAMETERS.MD)


## Run StackNet 

You can do so directly from the jar file, using Java higher than 1.6. You need to add Java as an environmental variable (e.g., add it to PATH).

The basic format is:

```
Java –jar stacknet.jar [train or predict] [task=regression or classification]  [parameter = value]
```

## Installations

This sections explains how to install the different external tools StackNet uses in its ensemble.

### Install Xgboost

Awesome xgboost can be used as a subprocess now in StackNet. This would require privileges to save and change files where the .jar is executed.

It is already pre-compiled for windows(64), mac and linux. 

**verify that the 'lib' folder os in the same directory where the StackNet.jar file is**. By default it should be there when you do `git clone`

for linux and mac you most probably need to change privileges for the executable : 
```
cd lib/
cd linux/
cd xg/
chmod +x xgboost
```
You can test that it works with :
`./xgboost`

It should print :

`Usage: <config>`

In windows and mac the behaviour should be similar. After executing `xgboost` from inside the `lib/your_operation_system/xg/` you should see the:

`Usage: <config>`

If you don't see this, then you need to compile it manually and drop the executables inside `lib/your_operation_system/xg/` .

You may find the follwing sources usefull:

-    [mac](https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_on_Mac_OSX?lang=en)
-    [windows1](https://stackoverflow.com/questions/33749735/how-to-install-xgboost-package-in-python-windows-platform) ,[windows2](https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en)
-    [linux](https://gist.github.com/DanielBeckstein/932087ee116cc2f72bfc6e3e078e899d)

Small Note: The user would need to delete the '.mod' files from inside the `model/` folder when no longer need them. StackNet does not do that automatically as it is not possible to determine when they are not needed anymore.  

**IMPORTANT NOTE:** This implementation does not include all Xgboost's features and the user is advised to use it directly from source to exploit its full potential. Also the version included is 6.0 and it is not certain whether it will be updated in the future as it required manual work to find all libraries and files required that need to be included for it to run. The performance and memory consumption will also be worse than running it directly from source. Additionally the descritpion of the parameters may not match the one in the offcial website, hence it is advised to use [xgboost's online parameter thread in github](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md) for more information about them.

### Install lightGBM

[lightGBM](https://github.com/Microsoft/LightGBM) can be used as a subprocess now in StackNet. This would require privileges to save and change files where the .jar is executed.

It is already pre-compiled for windows(64), mac and linux. 

**Verify that the 'lib' folder is in the same directory where the StackNet.jar file is**. By default it should be there when you do `git clone`

for linux and mac you most probably need to change privileges for the executable : 
```
cd lib/
cd linux/
cd lightgbm/
chmod +x lightgbm
```
You can test that it works with :
`./lightgbm`

It should print something in the form of:

```
[LightGBM] [Info] Finished loading parameters
[LightGBM] [Fatal] No training/prediction data, application quit
Met Exceptions:
No training/prediction data, application quit
```

In windows and mac the behaviour should be similar. After executing `lightgbm` from inside the `lib/your_operation_system/lightgbm/` you should see the:

`[LightGBM] [Info] Finished loading parameters...`

If you don't see this, then you need to compile it manually and drop the executables inside `lib/your_operation_system/lightgbm/` .

You may find the follwing sources usefull:

[Install LightGBM](https://github.com/Microsoft/LightGBM/wiki/Installation-Guide)

Small Note: The user would need to delete the '.mod' files from inside the `model/` folder when no longer need them. StackNet does not do that automatically as it is not possible to determine when they are not needed anymore.  

**IMPORTANT NOTE:** This implementation does not include all LightGBM's features and the user is advised to use it directly from source to exploit its full potential. it is not certain whether it will be updated in the future as it required manual work to find all libraries and files required that need to be included for it to run. The performance and memory consumption will also be worse than running it directly from source. Additionally the descritpion of the parameters may not match the one in the offcial website, hence it is advised to use [LightGBM's online parameter thread in github](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.md) for more information about them.

### Install H2O Algorithms

All the required jars are already packaged within the StackNet jar, however the user may find them inside the repo too. 

No special installation is required , but experimentally system protection might be blocking it , therefore make certain that the StackNet.jar is in the exceptions (firewall).

Additionally the first time StackNet uses an H2o Algorithm within the ensemble it takes more time (in comparison to every other time) because it sets up a cluster . 

### Install Fast_rgf

[fast_rgf](https://github.com/baidu/fast_rgf) can be used as a subprocess now in StackNet. This would require privileges to save and change files where the .jar is executed.

It is already pre-compiled for windows(64), mac and linux. 

**Verify that the 'lib' folder is in the same directory where the StackNet.jar file is**. By default it should be there when you do `git clone`

for linux and mac you most probably need to change privileges for the executable : 
```
cd lib/
cd linux/
cd frgf/
chmod +x forest_train
chmod +x forest_predict
```
You can test that it works with :
`./forest_train`

It should print something in the form of:

```
using up to x threads
```

In windows and mac the behaviour should be similar. After executing `forest_train` from inside the `lib/your_operation_system/frgf/` you should see the:

`using up to x threads...`

If you don't see this, then you need to compile it manually and drop the executables inside `lib/your_operation_system/frgf/` .

If you need to make the compilling manually for windows, you may find useful to download cmake fom :

[Install cmake](https://cmake.org/download/)

and use **mingw32-make.exe** as a compiler. 

Small Note: The user would need to delete the '.mod' files from inside the `model/` folder when no longer need them. StackNet does not do that automatically as it is not possible to determine when they are not needed anymore.  

**IMPORTANT NOTE:** This implementation does not include all fast_rgf's features and the user is advised to use it directly from source to exploit its full potential. it is not certain whether it will be updated in the future as it required manual work to find all libraries and files required that need to be included for it to run. The performance and memory consumption will also be worse than running it directly from source. Additionally the descritpion of the parameters may not match the one in the offcial website, hence it is advised to use [fast_rgf's online parameter thread in github](https://github.com/baidu/fast_rgf/tree/master/examples) for more information about them.

### Install Sklearn Algorithms

To install [Sklearn](http://scikit-learn.org/stable/) in StackNet you need **python higher-equal-to 2.7**. Python needs to be found in **PATH** as StackNet makes subprocesses in the command line. This would require privileges to save and change files where the .jar is executed.

**verify that the 'lib' folder is in the same directory where the StackNet.jar file is**

Once Python is installed and can be found on PATH, the user needs to isnstall **sklearn version 0.18.2** .

The following should do the trick in linux and mac. 

```
pip install scipy
pip install sklearn
```

For an easier installation in windows, the user could download [Anaconda](https://www.continuum.io/downloads) and make certain to check the **Add Anaconda's python to PATH** when it shows up during the installation.

All sklearn python scripts executed by StackNet are put in `lib/python/`

### Install Python Generic Algorithms

This a new feature that allows the user to run his/her own models as long as all libraries required can be found in his/her system when calling python. Assuming python is installed as explained in sklearn version above, the user may have a look inside **lib/python/**.

The scripts **PythonGenericRegressor0.py** and **PythonGenericClassifier0.py** are sample scripts that show how to format these models. The '0' is the main hyper parameter (called index) of the model PythonGenericRegressor (or PythonGenericClassifier). The data gets loaded in sparse format, but after this the user could add whetver he/she wants.

One could make many scritps and name them PythonGenericRegressor1,PythonGenericRegressor2...PythonGenericRegressorN and call them as:

```
PythonGenericRegressor index:1 seed:1 verbose:False 
PythonGenericRegressor index:2 seed:1 verbose:False 
PythonGenericRegressor index:N seed:1 verbose:False 
``` 
Once again **Verify that the 'lib' folder in the same directory where the StackNet.jar file is**.

### Install original libfm

[libFM](https://github.com/srendle/libfm) can be used as a subprocess now in StackNet. This would require privileges to save and change files where the .jar is executed.

It is already pre-compiled for windows(64), mac and linux. Note for windows libfm is compiled with [cygwin](https://www.cygwin.com/)

**Verify that the 'lib' folder is in the same directory where the StackNet.jar file is**. By default it should be there when you do `git clone`

for linux and mac you most probably need to change privileges for the executable : 
```
cd lib/
cd linux/
cd libfm/
chmod +x libfm
```
You can test that it works with :
`./libfm`

It should print something in the form of:

```
libFM
  Version: 1.4.2
   ...
   ...
```

In windows and mac the behaviour should be similar. After executing `libfm` from inside the `lib/your_operation_system/libfm/` you should see the same.

If you don't see this, **then you need to compile it manually** and drop the executables inside `lib/your_operation_system/libfm/`.

You may find the follwing sources usefull:

[libfm manual](http://www.libfm.org/libfm-1.42.manual.pdf)

**IMPORTANT NOTE:** This implementation may not include all libFM features plus it actually uses a version of it **that had a bug** on purpose. You can find more information about why this was chosen in the following [python wrapper for libFM](https://github.com/jfloff/pywFM). It basically had this bug that was allowing you to get the parameters of the trained models for all training methods. These parameters are now extracted once a model has been trained and the scoring uses only these parameters (e.g. not the libFM executable). 

Also, multiclass problems are formed as binary 1-vs-all.

Bear in mind the [licence of libfm](https://github.com/srendle/libfm/blob/master/license.txt). If you find it useful, cite the following paper :
Rendle, S. (2012). Factorization machines with libfm. *ACM Transactions on Intelligent Systems and Technology (TIST)*, 3(3), 57.Chicago .  [Link](http://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf) 

### Install vowpal wabbit

[vowpal wabbit](https://github.com/JohnLangford/vowpal_wabbit) can be used as a subprocess now in StackNet. This would require privileges to save and change files where the .jar is executed.

It is already pre-compiled for windows(64) and linux.

**Mac** was more difficult than expected and generally there is a lack of expertise working with Mac. If someone could help here, please email me at kazanovassoftware@gmail.com. 

For mac, you have to install vowpal wabbit from source and drop the executable in `lib/mac/vw/`. Consider [the following link](https://github.com/JohnLangford/vowpal_wabbit#mac-os-x-specific-info). `brew install vowpal-wabbit` will most probably do the trick.
If that does not work, you may execute the`lib/mac/vw/script.sh`. This is not advised though as it will override some files you may have already installed - use it as a last resort.

**Verify that the 'lib' folder is in the same directory where the StackNet.jar file is**. By default it should be there when you do `git clone`

for linux and mac you most probably need to change privileges for the executable : 
```
cd lib/
cd linux/
cd vw/
chmod +x vw
```
You can test that it works with :
`./vw`

It should print something in the form of:

```
	Num weight bits = 18
	learning rate = 0.5
	initial_t = 0
    ...
```

In windows and mac the behaviour should be similar. After executing `vw` from inside the `lib/your_operation_system/vw/` you should see the same.

If you don't see this, **then you need to compile it manually** and drop the executables inside `lib/your_operation_system/vw/`.

You may find the follwing sources usefull:

[Download suggestions](https://github.com/JohnLangford/vowpal_wabbit/wiki/Download)

**IMPORTANT NOTE:** This implementation may not include all Vowpal Wabbit features and the user is advised to use it directly from the source. Also the version may not be the final and it is not certain whether it will be updated in the future as it required manual work to find all libraries and files required that need to be included for it to run. The performance and memory consumption will also be worse than running directly. Additionally the descritpion of the parameters may not match the one in the website, hence it is advised to use [VW's online parameter thread in github](https://github.com/JohnLangford/vowpal_wabbit/wiki/Command-line-arguments) for more information about them.


### Install libffm

[libffm](https://github.com/guestwalk/libffm) can be used as a subprocess now in StackNet. This would require privileges to save and change files where the .jar is executed.

It is already pre-compiled for windows(64), mac and linux. 

**Verify that the 'lib' folder is in the same directory where the StackNet.jar file is**. By default it should be there when you do `git clone`

for linux and mac you most probably need to change privileges for the executable : 
```
cd lib/
cd linux/
cd libffm/
chmod +x ffm-train
chmod +x ffm-predict
```
You can test that it works with :
`./ffm-train`

It should print something in the form of:

```
usage: ffm-train [options] training_set_file [model_file]

options:
-l <lambda>: set regularization parameter (default 0.00002)
-k <factor>: set number of latent factors (default 4)
-t <iteration>: set number of iterations (default 15)
```
You should also test that it works with :
`./ffm-predict`

It should print:

```
usage: ffm-predict test_file model_file output_file
```

In windows and mac the behaviour should be similar. After executing `ffm-train` or `ffm-predict` from inside the `lib/your_operation_system/libffm/` you should see the same results.

If you don't see this, then you need to compile it manually and drop the executables inside `lib/your_operation_system/libffm/` .

You may find the follwing sources usefull:

[Install libffm](https://github.com/guestwalk/libffm) . Search for `Installation ...` and `OpenMP and SSE ...`

Small Note: The user would need to delete the '.mod' files from inside the `model/` folder when no longer need them. StackNet does not do that automatically as it is not possible to determine when they are not needed anymore.  

**IMPORTANT NOTE:** This implementation may not include all libffm features and the user is advised to use it directly from the source. Also the version may not be the final and it is not certain whether it will be updated in the future as it required manual work to find all libraries and files required that need to be included for it to run. The performance and memory consumption will also be worse than running directly . Additionally the descritpion of the parameters may not match the one in the website, hence it is advised to use libffm online parameter thread in github for more information about them. 
Also, multiclass problems are formed as binary 1-vs-all.


## Command Line Parameters

Command | Explanation
--- | ---
task  | could be either **regression** or **classification**.</li>
sparse  | True if the data to be imported are in sparse format (libsvm) or dense (false) 
has_head   | True if train_file and test_file have headers else false
model | Name of the output model file. 
pred_file | Name of the output prediction file. 
train_file | Name of the training file. 
test_file | Name of the test file. 
output_name | Prefix of the models to be printed per iteration. This is to allow the Meta features of each iteration to be printed. Defaults to nothing.
data_prefix |  prefix to be used when the user supplies own pairs of [X_train,X_cv] datasets for each fold as well as an X file for the whole training data. This is particularly useful for when likelihood features are needed or generally features than must be computed within cv.  Each train/valid pair is identified by prefix_train[fold_index_starting_from_zero].txt/prefix_cv[fold_index_starting_from_zero].txt and prefix_train.txt for the final set. For example if prefix=mystack and folds=2 then stacknet is expecting 2 pairs of train/cv files. e.g [[mystack_train0.txt,mystack_cv0.txt],[mystack_train1.txt,mystack_cv1.txt]]. It also expects a [mystack_train.txt]  for the final train set. These files can be either dense or sparse ( when 'sparse=True') and need to have the target variable in the beginning. If you use **output_name** to extract the predictions, these will be stacked vertically in the same order as the cv files. 
indices_name |  A prefix. When given any value it prints a .csv file for each fold with the corresponding train(0) and valiation(1) indices stacked vertically .The format is “row_index,[0 if train else 1 for validation]”. First it prints the train indices and then the validation indices in exactly the same order as they appear when modelling inside StackNet.
input_index (**New**) |  Name of file to load in order to form the train and cv indices during kfold cross validation. This overrides the internal process for generating kfolds and ignores the given folds. Each row needs to contain an integer in that file. Row size of the file needs to be the same as the `train_file`. It should not contain headers. one line=one integer - the indice of the validation fold the case belongs to.[There is an example](/example/manual_index/EXAMPLE.MD)
include_target (**New**) |  True to enable printing the target column in the output file for train holdout predictions (when `output_name` is not empty).
test_target | True if the test file has a target variable in the beginning (left) else false (only predictors in the file).
params | Parameter file where each line is a model. empty lines correspond to the creation of new levels 
verbose | True if we need StackNet to output its progress else false 
threads | Number of models to run in parallel. This is independent of any extra threads allocated from the selected algorithms. e.g. it is possible to run 4 models in parallel where one is a randomforest that runs on 10 threads (it selected). 
metric | Metric to output in cross validation for each model-neuron. can be logloss, accuracy or auc (for binary only) for classification and rmse ,rsquared or mae for regerssion .defaults to 'logloss' for classification and 'rmse' for regression. 
stackdata | True for restacking else false
seed | Integer for randomised procedures 
bins | A parameter that allows classifiers to be used in regression problems. It first bins (digitises) the target variable and then runs classifiers on the transformed variable. Defaults to 2</li>. 
folds | Number of folds for re-usable kfold

### Parameters' File

In The parameter file, each line is a model. When there is an empty line then any new algorithm is used in the next level.  This is a sample format. 
Note this file accepts comments (`#`). Anything on the right of the `#` symbol is ignored.(**New**)

```
LogisticRegression C:1 Type:Liblinear maxim_Iteration:100 scale:true verbose:false
RandomForestClassifier bootsrap:false estimators:100 threads:5 logit.offset:0.00001 verbose:false cut_off_subsample:1.0 feature_subselection:1.0 gamma:0.00001 max_depth:8 max_features:0.25 max_tree_size:-1 min_leaf:2.0 min_split:5.0 Objective:ENTROPY row_subsample:0.95 seed:1
GradientBoostingForestClassifier estimators:100 threads: offset:0.00001 verbose:false trees:1 rounding:2 shrinkage:0.05 cut_off_subsample:1.0 feature_subselection:0.8 gamma:0.00001 max_depth:8 max_features:1.0 max_tree_size:-1 min_leaf:2.0 min_split:5.0 Objective:RMSE row_subsample:0.9 seed:1
Vanilla2hnnclassifier UseConstant:true usescale:true seed:1 Type:SGD maxim_Iteration:50 C:0.000001 learn_rate:0.009 smooth:0.02 h1:30 h2:20 connection_nonlinearity:Relu init_values:0.02
LSVC Type:Liblinear threads:1 C:1.0 maxim_Iteration:100 seed:1
LibFmClassifier lfeatures:3 init_values:0.035 smooth:0.05 learn_rate:0.1 threads:1 C:0.00001 maxim_Iteration:15 seed:1
NaiveBayesClassifier usescale:true threads:1 Shrinkage:0.1 seed:1 verbose:false
XgboostRegressor booster:gbtree objective:reg:linear num_round:100 eta:0.015 threads:1 gamma:2.0 max_depth:4 subsample:0.8 colsample_bytree:0.4 seed:1 verbose:false
XgboostRegressor booster:gblinear objective:reg:gamma num_round:500 eta:0.5 threads:1 lambda:1 alpha:1 seed:1 verbose:false

RandomForestClassifier estimators=1000 rounding:3 threads:4 max_depth:6 max_features:0.6 min_leaf:2.0 Objective:ENTROPY gamma:0.000001 row_subsample:1.0 verbose:false copy=false
```

**Tip**: To tune a single model, one may choose an algorithm for the first layer and a dummy one for the second layer. StackNet expects at least two algorithms, so with this format the user can visualize the performance of single algorithm inside the K-fold.
For example, if I wanted to tune a Random Forest Classifier, I would put it in the first line (layer) and also put any model (lets say Logistic Regression) in the second layer and could break the process immediately after the first layer kfold is done:

```
RandomForestClassifier bootsrap:false estimators:100 threads:5 logit.offset:0.00001 verbose:false cut_off_subsample:1.0 feature_subselection:1.0 gamma:0.00001 max_depth:8 max_features:0.25 max_tree_size:-1 min_leaf:2.0 min_split:5.0 Objective:ENTROPY row_subsample:0.95 seed:1

LogisticRegression verbose:false
```

## Data Format

For **dense** input data, the file needs to start with the target variable followed by a comma, separated variables like:

1,0,0,2,3,2.4

0,1,1,0,0,12

For **sparse** format , it is the same as libsvm (same example as above)  :

1 2:2 3:3 4:2.4

0 0:1 1:1 4:12

**warning**: Some algorithms (mostly tree-based) may not be very fast with this format)

If test_target is false, then the test data may not have a target and start directly from the variables.

A **train** method needs at least a **train_file** and a **params_file**. It also needs at least two algorithms, and the and last layer must not contain a regressor unless the metric is auc and the problem is binary.

A **predict** method needs at least a **test_file** and a **model_file**. 



## Commandline Train Statement
Java –jar stacknet.jar **_train_** **task=classification** **sparse**=false **has_head**=true **model**=model **pred_file**=pred.csv **train_file**=sample_train.csv **test_file**= sample_test.csv **test_target**=true **params**=params.txt **verbose**=true **threads**=7 **metric**=logloss **stackdata**=false **seed**=1 **folds**=5 **bins**=3

Note that you can have train and test at the same time. In that case after training, it scores the test data. 

## Commandline predict Statement

Java -jar stacknet.jar **_predict_** **sparse**=false **has_head**=true **model**=model **pred_file**=pred.csv **test_file**=sample_test.csv **test_target**=true **verbose**=true  **metric**=logloss


## Examples

- [Kaggle-Quora-sparse](/example/Quora_kaggle_sparse/README.MD)
- [Kaggle-TwoSigma](/example/twosigma_kaggle/EXAMPLE.MD)
- [Kaggle-TwoSigma Random Forest using the Library](/example/twosigma_kaggle_java_rf/EXAMPLE.MD)
- [Kaggle-Amazon Classification challenge and use of data_prefix](/example/example_amazon/EXAMPLE.MD)
- [Kaggle-Zillow regerssion example](/example/zillow_regression_sparse/README.MD)
- [Example_with_index](/example/manual_index/EXAMPLE.MD)




## Run StackNet from within Java code


If we wanted to build a 3-level stacknet on a binary target with desne data, we start with initializing a _StackNetClassifier_ Object:

```java
 StackNetClassifier StackNet = new StackNetClassifier (); // Initialise a StackNet 
``` 

Which is then followed by a 2-dimensional String array with the list of models in each layer along with their hyperparameters in the form of as in "_estimator [space delimited hyper parameters]_"

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

Ultimately given a data object X and a 1-dimensional vector y, the model can be trained using:

```java
StackNet.target=y; // the target variable        
StackNet.fit(X); // fitting the model on the training data

```
Predictions are made with :

```java
double preds [][]=StackNet.predict_proba(X_test);
```

## Potential Next Steps

- ~~Add StackNetRegressor~~ Done. 
- ~Add [H<sub>2</sub>O](http://h2o-release.s3.amazonaws.com/h2o/master/3908/index.html)~
- ~increase coverage in general with well-known and well-performing ml tools (original libfm, libffm, vowpal wabbit)~
- Add data pre-processing steps
- Make a python wrapper

## Reference

For now, you may use this:

Marios Michailidis (2017), StackNet, StackNet Meta Modelling Framework, url https://github.com/kaz-Anova/StackNet

## News

- StackNet model was presented at [infiniteconf 2017](https://skillsmatter.com/conferences/7983-infiniteconf-2017-the-conference-on-big-data-data-science-and-engineering#program ) [6th-7th July] and the video is available there if you sign up
- New [facebook page](https://www.facebook.com/StackNet/) to discuss StackNet and other open source data science topics.
- StackNet and Sracking was explained in [kaggle's blog](http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/)
- The is an Ask Me Anything (AMA) [thread in kaggle](https://www.kaggle.com/general/34802) with useful material about stacking and StackNet.
- A workshop with StackNet will take place in [ODSC in London](https://www.odsc.com/london/speakers) October 12-14 .

## Special Thanks

To my co-supervisors:

- [Giles Pavey](https://www.linkedin.com/in/giles-pavey-924426/)
- [Prof. Philip Treleaven](http://www0.cs.ucl.ac.uk/staff/P.Treleaven/) 

