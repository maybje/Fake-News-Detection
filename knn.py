#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from pandas import DataFrame as df
from sklearn.model_selection import (KFold, cross_val_score,
                                    GridSearchCV)
from scipy import stats
import seaborn as sns
from sklearn.metrics import (confusion_matrix,cohen_kappa_score,recall_score,
                    precision_score)
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

#Set wd
os.chdir("D:\\Documentos\\Essex\\Machine Learning\\assignment")
cwd=os.getcwd()
#read data
data=pd.read_csv("data\\train_imp.csv", header=0)
#print(data.head())

#create feature matrix and feature vectors
#Training set
y=data.iloc[:,-1]
x=data.iloc[:,:-1]
names=list(x.columns)
print("Shape X matrix: ", x.shape)
print("prop: ", y.value_counts()/y.shape[0])

#validation set
data2=pd.read_csv("data\\val_imp.csv", header=0)
y_v=data2.iloc[:,-1]
x_v=data2.iloc[:,:-1]
print("Shape X_v matrix: ", x_v.shape)

#############################
#Feature selection
#############################
#setting up feature selection algorithm
k_fold = KFold(n_splits=10)
est = LogisticRegression()
selector=RFECV(est,cv=k_fold)
selector.fit(x,y)
#keeping selected variables and printing names for control
x_b=x.loc[:,selector.get_support()]
xv_b=x_v.loc[:,selector.get_support()]
print("Optimal number of features : %d" % selector.n_features_)
print("Support", x.loc[:,selector.get_support()].columns)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.savefig("plots\\featknn.pdf", bbox_inches='tight')
plt.close()

##############################
#KNN
##############################
#Setting up classifiers and parameter grid
nca = NeighborhoodComponentsAnalysis(n_components=9)
knn = KNeighborsClassifier(n_neighbors=4)
nn=np.arange(2,15,2)

#defining pipeline
pipe = Pipeline([('nca', nca),
                 ('knn', knn)])

#gridsearch
gs = GridSearchCV(estimator=pipe,\
    param_grid=dict(knn__n_neighbors=nn),\
    n_jobs=-1, cv=k_fold)

#Fitting and printing cv accuracy
gs.fit(x_b,y)
print('Best nn:', gs.best_estimator_)
print('average score:', gs.best_score_)

#Test accuracy and other measures
y_pred=gs.predict(xv_b)
print(y_pred[:10])
kappa=cohen_kappa_score(y_v,y_pred)
print("Kappa: ", kappa)
print("Score: ", gs.score(xv_b,y_v))
print("Recall: ", recall_score(y_v,y_pred))
print("Precision: ", precision_score(y_v,y_pred))
print("confussion: ", confusion_matrix(y_v,y_pred))
