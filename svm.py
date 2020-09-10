#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from pandas import DataFrame as df
from sklearn import svm
from sklearn.model_selection import (KFold, cross_val_score,
                                    GridSearchCV)
from scipy import stats
import seaborn as sns
from sklearn.metrics import (confusion_matrix,cohen_kappa_score,recall_score,
                    precision_score)
from sklearn.feature_selection import RFECV

#Set wd
os.chdir("D:\\Documentos\\Essex\\Machine Learning\\assignment")
cwd=os.getcwd()
#read data
data=pd.read_csv("data\\train_imp.csv", header=0)
data3=pd.read_csv("data\\test_imp.csv", header=0)
#print(data.head())

#create feature matrix and feature vectors
#Training set
y=data.iloc[:,-1]
x=data.iloc[:,:-1]
names=list(x.columns)
print("Shape X matrix: ", x.shape)
print("prop: ", y.value_counts()/y.shape[0])

#Validation set
data2=pd.read_csv("data\\val_imp.csv", header=0)
y_v=data2.iloc[:,-1]
x_v=data2.iloc[:,:-1]
print("Shape X_v matrix: ", x_v.shape)

#test set
y_t=data3.iloc[:,-1]
x_t=data3.iloc[:,:-1]
print("Shape X_t matrix: ", x_t.shape)


#############################
#Feature selection
#############################
#setting up feature selection algorithm
k_fold = KFold(n_splits=10)
est=svm.SVC(kernel="linear", random_state=21)
selector=RFECV(est,cv=k_fold)
selector.fit(x,y)
#keeping selected variables and printing names for control
x=x.loc[:,selector.get_support()]
x_v=x_v.loc[:,selector.get_support()]
print("Optimal number of features : %d" % selector.n_features_)
print("Support", x.columns)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.savefig("plots\\featsvm.pdf", bbox_inches='tight')
plt.close()

#############
#svm
#############
#Setting up algorithm and parameter grid
clf = svm.SVC(gamma=0.001, C=1000.,kernel="rbf", random_state=21)
Cs = np.logspace(-1, 3, 9)
Gs = np.logspace(-7, -0, 9)

clf = GridSearchCV(estimator=clf, \
    param_grid=dict(C=Cs, gamma=Gs), n_jobs=-1, cv=k_fold)

#Fitting and printing cv accuracy
clf.fit(x,y)
print('Best C:', clf.best_estimator_.C,
        'Best gamma:', clf.best_estimator_.gamma)
print('best accuracy:', clf.best_score_)

#Test accuracy and other measures
y_pred=clf.predict(x_v)
kappa=cohen_kappa_score(y_v,y_pred)
print("Score: ", clf.score(x_v,y_v))
print("Kappa: ", kappa)
print("Recall: ", recall_score(y_v,y_pred))
print("Precision: ", precision_score(y_v,y_pred))
print("confussion: ", confusion_matrix(y_v,y_pred))

x_t=x_t.loc[:,selector.get_support()]
#Predicting classes
y_test=clf.predict(x_t)

#Saving Results
pd.DataFrame(y_test, columns=["Class"]).to_csv("data\\tets_svm.csv", index=False)
print("test results shape: ", pd.DataFrame(y_test, columns=["Class"]).shape)
