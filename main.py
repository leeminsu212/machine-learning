import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# according to search below set ? as null data
df=pd.read_csv('C:/Users/leeminsu/PycharmProjects/breastCancer/breast-cancer-wisconsin.data', na_values='?')
col=['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# # search what kind of data in dataframe
# for i in range(1, 11, 1):
#     print(col[i])
#     print(df[[col[i]]].groupby([col[i]]).sum())
#     print()

# drop ID column and rows that include ? as data
df.dropna(inplace=True)
df=df.drop(['ID'], axis=1)

x=df.drop(['Class'], axis=1)
y=df.pop('Class')

# function make best scaled dataframe
# this function show best accuracy and scaler for classifier
# user can search best scaler for classifier
# user have to input data(x), target(y) and classifier
def makeCombi(x ,y, classifier):
    nameScaler=['StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler']
    scaler=[preprocessing.StandardScaler(), preprocessing.MinMaxScaler(), preprocessing.MaxAbsScaler(), preprocessing.RobustScaler()]
    # listDf is list to store scaled dataframes
    listScaledDf=[]
    col=x.columns.values

    # scaling by each scaler and store them
    for i in range(len(scaler)):
        sc=scaler[i]
        tempX=sc.fit_transform(x)
        tempX=pd.DataFrame(tempX, columns=col)
        listScaledDf.append(tempX)

    # search best scaler for classifier
    print('Classifier :', classifier)
    scoreBest = 0
    indexBest = 0
    for i in range(len(listScaledDf)):
        trainSetX, testSetX, trainSetY, testSetY = train_test_split(listScaledDf[i], y, test_size=0.2)
        classifier.fit(trainSetX, trainSetY)
        score=classifier.score(testSetX, testSetY)
        print('Scaler :', nameScaler[i], 'Score :', score)
        if(scoreBest<=score):
            scoreBest=score
            indexBest=i
    print('\nBest scaler for', classifier)
    print('Scaler :', nameScaler[indexBest], 'Score :', scoreBest, '\n')

    # user can get dataframe scaled with scaler that make best accuracy
    # output is the best dataframe
    return listScaledDf[indexBest]

# function evaluate each model
# input data and target, classifier
def evaluation(x, y, classifier, k):
    trainSetX, testSetX, trainSetY, testSetY = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1)
    score=cross_val_score(classifier, trainSetX, trainSetY, cv=k)
    print(classifier, '\nCross validation score :', score)
    print('Mean score :', score.mean())
    classifier.fit(trainSetX, trainSetY)
    print('Accuracy on test set :', classifier.score(testSetX, testSetY))

# ------------decision tree(entropy)------------
bestXdte=makeCombi(x, y, DecisionTreeClassifier(criterion='entropy'))

# try various max depth and max features for decision tree(entropy)
trainSetX, testSetX, trainSetY, testSetY = train_test_split(bestXdte, y, test_size=0.2, shuffle=True, random_state=1)
param_grid=[{'max_depth':np.arange(1, 21), 'max_features':np.arange(1, len(bestXdte.columns) + 1)}]
dteGridSearchCV=GridSearchCV(DecisionTreeClassifier(criterion='entropy'), param_grid, cv=5)
dteGridSearchCV.fit(trainSetX, trainSetY)
print(dteGridSearchCV.best_params_)
print('Best score :', dteGridSearchCV.best_score_)

# store best max_depth value and max_features value
bestDepth=dteGridSearchCV.best_params_.get('max_depth')
bestFeatures=dteGridSearchCV.best_params_.get('max_features')

# evaluate decision tree(entropy)
dte=DecisionTreeClassifier(criterion='entropy', max_depth=bestDepth, max_features=bestFeatures)
# try various value of k for k-fold CV
for k in range(2, 10, 2):
    print('\nCross validation with k=', k)
    evaluation(bestXdte, y, dte, k)

# ------------decision tree(gini)------------
bestXdtg=makeCombi(x, y, DecisionTreeClassifier(criterion='gini'))

# try various max depth and max features for decision tree(gini)
trainSetX, testSetX, trainSetY, testSetY = train_test_split(bestXdtg, y, test_size=0.2, shuffle=True, random_state=1)
param_grid=[{'max_depth':np.arange(1, 21), 'max_features':np.arange(1, len(bestXdtg.columns) + 1)}]
dtgGridSearchCV=GridSearchCV(DecisionTreeClassifier(criterion='gini'), param_grid, cv=5)
dtgGridSearchCV.fit(trainSetX, trainSetY)
print(dtgGridSearchCV.best_params_)
print('Best score :', dtgGridSearchCV.best_score_)

# store best max_depth value and max_features value
bestDepth=dtgGridSearchCV.best_params_.get('max_depth')
bestFeatures=dtgGridSearchCV.best_params_.get('max_features')

# evaluate decision tree(gini)
dtg=DecisionTreeClassifier(criterion='gini', max_depth=bestDepth, max_features=bestFeatures)
# try various value of k for k-fold CV
for k in range(2, 10, 2):
    print('\nCross validation with k=', k)
    evaluation(bestXdtg, y, dtg, k)

# ------------logistic regression------------
bestXlr=makeCombi(x, y, LogisticRegression())

# try various solver for logistic regression
trainSetX, testSetX, trainSetY, testSetY = train_test_split(bestXlr, y, test_size=0.2, shuffle=True, random_state=1)
param_grid=[{'solver':['lbfgs', 'sag', 'saga'], 'warm_start':[False, True]}]
lrGridSearchCV=GridSearchCV(LogisticRegression(), param_grid, cv=5)
lrGridSearchCV.fit(trainSetX, trainSetY)
print(lrGridSearchCV.best_params_)
print('Best score :', lrGridSearchCV.best_score_)

# store best max_depth value and max_features value
bestSolver=lrGridSearchCV.best_params_.get('solver')
bestWarmStart=lrGridSearchCV.best_params_.get('warm_start')

# evaluate logistic regression
lr=LogisticRegression(solver=bestSolver, warm_start=bestWarmStart)
# try various value of k for k-fold CV
for k in range(2, 10, 2):
    print('\nCross validation with k=', k)
    evaluation(bestXlr, y, lr, k)

# ------------SVM------------
bestXsvm=makeCombi(x, y, SVC())

# try various C and kernel for SVM
trainSetX, testSetX, trainSetY, testSetY = train_test_split(bestXsvm, y, test_size=0.2, shuffle=True, random_state=1)
param_grid=[{'C':[0.01, 0.1, 1.0, 10.0], 'kernel':['linear', 'poly', 'rbf']}]
svmGridSearchCV=GridSearchCV(SVC(), param_grid, cv=5)
svmGridSearchCV.fit(trainSetX, trainSetY)
print(svmGridSearchCV.best_params_)
print('Best score :', svmGridSearchCV.best_score_)

# store best C value and kernel value
bestC=svmGridSearchCV.best_params_.get('C')
bestKernel=svmGridSearchCV.best_params_.get('kernel')

# evaluate SVM
svm=SVC(C=bestC, kernel=bestKernel)
# try various value of k for k-fold CV
for k in range(2, 10, 2):
    print('\nCross validation with k=', k)
    evaluation(bestXsvm, y, svm, k)
