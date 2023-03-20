# Machine learning PHW

Data from uci.edu https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

Compare performance of multiple classification models.</br>
#### Classification models
* Decision tree(entropy)
* Decision tree(gini)
* Logistic regression
* SVM

For each model compare 4 scalers(Standard scaler, MinMax scaler, MaxAbs scaler, Robust scaler) and find the best scaler that return the best accuracy. Then find the best hyperparameters using GridSearchCV(cv=5). </br>
#### Hyperparameters
* Decision tree(entropy, gini) - max_depth(1-20), max_features(1-9)
* Logistic regression - solver, dual, warm_start
* SVM - C, kernel

Finally evaluate each model using k-fold CV. I compared 4 k values(2, 4, 6, 8).

## Result

### Decision tree(entropy)
![image](https://user-images.githubusercontent.com/33173280/226265653-a32f9dc1-553f-492b-a049-195e7f943353.png)</br>
Best scaler is Robust scaler.</br>
#### Best hyperparameters</br>
![image](https://user-images.githubusercontent.com/33173280/226264299-14682b7a-2c6c-46d7-91fd-5129aa715d52.png)
![image](https://user-images.githubusercontent.com/33173280/226276919-50f72db1-9e47-48f0-8d47-47b3ac38eaed.png)</br>
The best accuracy is when k = 8

### Decision tree(gini)
![image](https://user-images.githubusercontent.com/33173280/226266380-777f6818-f017-42dd-8a44-defb276f3fab.png)</br>
Best scaler is Robust scaler.</br>
#### Best hyperparameters</br>
![image](https://user-images.githubusercontent.com/33173280/226264414-455bcf8d-6fc2-49fa-b102-b96f4c1a5ae2.png)
![image](https://user-images.githubusercontent.com/33173280/226266516-17af0a27-253e-402d-9418-f55c6c63323e.png)</br>
The best accuracy is when k = 4
