[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13182499&assignment_repo_type=AssignmentRepo)
# Auto-sklearn Fail

The code in `fail.py` runs
[auto-sklearn](https://automl.github.io/auto-sklearn/master/) on a dataset for 5
minutes. The model it finds after that time is *worse* than just using a random
forest with default hyperparameters.

Find out what's going on, why auto-sklearn's performance is so bad, and how to
fix it.


# Explanation

The core of the problem is the encoding done by OneHotEncoder. The OneHotEncoder is meant for use on categorical data. The data it is being applied to is very much not categorical. So it makes sense that the performance is very poor. By simply commenting out those lines, the performance of the auto-sklearn model jumps up to be nearly as good as the RF default (off by .01). So what else can we adjust? 


Tripling the time for the auto-sklearn model raised its performance but still not enough to beat the default RF model. So the easy fix is out. 


Taking a look at how the automl model is currently performing reveals that it is grossly overfitting. The auto-sklearn classifier doesn't appear to have a method for detecting/preventing overfitting so the best I can do is have it do data resampling. The option I tried was 'cv' for cross validation and set it to do 10 folds (no particular reason, just seemed like a good number). 

This seems to work and brings the automl model past the random forest model:

RandomForest Accuracy 0.67
AutoML Accuracy 0.6725
AutoML Train Accuracy 1.0

Doesn't seem to actually fix the overfitting problem, though. Just makes it better. Now that it least performs a little bit better, we could then increase the runtime to see more improvements. Running it for 30 minutes instead of 5 gets the accuracy to:

AutoML Accuracy 0.675

Not much better really
