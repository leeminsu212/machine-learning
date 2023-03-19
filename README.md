# Machine learning PHW

Data from uci.edu https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

Compare performance of multiple classification models.
* Decision tree(entropy)
* Decision tree(gini)
* Logistic regression
* SVM

For each model compare 4 scalers(Standard scaler, MinMax scaler, MaxAbs scaler, Robust scaler) and find the best scaler that return the best accuracy. Then find the best hyperparameters.</br>
#### Hyperparameters
* Decision tree(entropy, gini) - max_depth, max_features
* Logistic regression - solver, dual, warm_start
* SVM - C, kernel

Finally evaluate each model using k-fold CV. I compared 4 k values(2, 4, 6, 8).
