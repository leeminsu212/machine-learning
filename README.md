# Machine learning PHW

Data from uci.edu https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

Compare performance of multiple classification models.</br>
#### Classification models
* Decision tree(entropy)
* Decision tree(gini)
* Logistic regression
* SVM

For each model compare 4 scalers(Standard scaler, MinMax scaler, MaxAbs scaler, Robust scaler) and find the best scaler that return the best accuracy. Then find the best hyperparameters.</br>
#### Hyperparameters
* Decision tree(entropy, gini) - max_depth(1-20), max_features(1-9)
* Logistic regression - solver, dual, warm_start
* SVM - C, kernel

Finally evaluate each model using k-fold CV. I compared 4 k values(2, 4, 6, 8).

### Result

##### Decision tree(entropy)
<img src="https://user-images.githubusercontent.com/33173280/226260034-7a76a1de-2d9d-41fb-a2d3-c03ca495b4f1.png" width="500" height="200">

##### Decision tree(gini)
<img src="https://user-images.githubusercontent.com/33173280/226260214-9f212d52-a31d-464e-93ab-e55e52d68fac.png" width="500" height="200">
