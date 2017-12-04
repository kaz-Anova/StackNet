# Example For The H2O WORLD

This example shows the benefits of multi-layer stacking on some artificial data. 

The Example inlcudes a *light* version of StackNet that includes only the native algorithms in order to avoid installation problems (that may appear when installing the wrappers).

This example is illustrated in the powerpoint, if you view it **as a slideshow**. This is IMPORTANT otherwise the slide will not make sense since it contains animations. 

To run follow the next steps:

1. From the command line *cd* to the location of where this example resides.  
2. Architecture of a linear base learner and a linear meta learner **with fairly untuned parameters**: Run `java -jar StackNet.jar train task=regression train_file=train.csv test_file=test.csv params=params.txt pred_file=pred.csv test_target=true has_head=true verbose=true Threads=3 folds=4 metric=rmse`. This will produce test predictions in *pred.csv*.
3. Architecture of a linear base learner and a linear meta learner **with tuned parameters**: Run `java -jar StackNet.jar train task=regression train_file=train.csv test_file=test.csv params=params0.txt pred_file=pred0.csv test_target=true has_head=true verbose=true Threads=3 folds=4 metric=rmse`. This will produce test predictions in *pred0.csv*.
4. Architecture of a linear base learner, a Gradient Boosting Machine base learner and a linear meta learner: Run `java -jar StackNet.jar train task=regression train_file=train.csv test_file=test.csv params=params1.txt pred_file=pred1.csv test_target=true has_head=true verbose=true Threads=3 folds=4 metric=rmse`. This will produce test predictions in *pred1.csv*.
5. Architecture of a linear base learner, a Gradient Boosting Machine base learner, a Multi-Layer Perceptron base learner and a linear meta learner: Run `java -jar StackNet.jar train task=regression train_file=train.csv test_file=test.csv params=params2.txt pred_file=pred2.csv test_target=true has_head=true verbose=true Threads=3 folds=4 metric=rmse`. This will produce test predictions in *pred2.csv*.
6. Architecture of a linear base learner, a Gradient Boosting Machine base learner, a Multi-Layer Perceptron base learner and a Random Forest meta learner: Run `java -jar StackNet.jar train task=regression train_file=train.csv test_file=test.csv params=params3.txt pred_file=pred3.csv test_target=true has_head=true verbose=true Threads=3 folds=4 metric=rmse`. This will produce test predictions in *pred3.csv*.
7. Architecture of a linear base learner, a Gradient Boosting Machine base learner, a Multi-Layer Perceptron base learner, a Random Forest meta learner, a linear meta learner and a linear level 2 (meta meta) leanrner :Run `java -jar StackNet.jar train task=regression train_file=train.csv test_file=test.csv params=params4.txt pred_file=pred4.csv test_target=true has_head=true verbose=true Threads=3 folds=4 metric=rmse`. This will produce test predictions in *pred4.csv*.

