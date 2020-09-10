#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
np.random.seed(21)
from tensorflow import keras
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from pandas import DataFrame as df
from sklearn import svm
from sklearn.model_selection import (KFold, cross_val_score,
                                    GridSearchCV)
from sklearn.feature_selection import RFECV
from sklearn.metrics import (confusion_matrix,cohen_kappa_score,recall_score,
                    precision_score)
os.chdir("D:\\Documentos\\Essex\\Machine Learning\\assignment")
cwd=os.getcwd()
#read data
data=pd.read_csv("data\\train_imp.csv", header=0)
data2=pd.read_csv("data\\val_imp.csv", header=0)
data3=pd.read_csv("data\\test_imp.csv", header=0)
#print(data.head())

#create feature matrix and feature vectors
#training set
y=data.iloc[:,-1]
x=data.iloc[:,:-1]
names=list(x.columns)
print("Shape X matrix: ", x.shape)
print(y.head())

#Validation set
y_v=data2.iloc[:,-1]
x_v=data2.iloc[:,:-1]
print("Shape X_v matrix: ", x_v.shape)

x=x.append(x_v,ignore_index=True)
y=y.append(y_v, ignore_index=True)

#Test set
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


########################
#ANN
########################
#Setting up nerual network and estimating CV accuracy
##2 hidden layers with 45 and 5 neurons, respectively
score=[]
for train_indices, test_indices in k_fold.split(x):
    model=keras.Sequential([
        keras.layers.Input(shape=selector.n_features_.item()),
        keras.layers.Dense(45,activation="relu",kernel_initializer='orthogonal'),
        keras.layers.Dense(5,activation="relu",kernel_initializer='orthogonal'),
        keras.layers.Dense(2, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", \
        metrics=["accuracy"])
    model.fit(x.loc[train_indices], y.loc[train_indices],epochs=40, batch_size=10)
    _,test_acc=model.evaluate(x.loc[test_indices],y.loc[test_indices])
    score.append(test_acc)
print('Average accuracy:', np.mean(score))

#training neural network
model=keras.Sequential([
    keras.layers.Input(shape=selector.n_features_.item()),
    keras.layers.Dense(45,activation="relu",kernel_initializer='orthogonal'),
    keras.layers.Dense(5,activation="relu",kernel_initializer='orthogonal'),
    keras.layers.Dense(2, activation="sigmoid")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", \
    metrics=["accuracy"])

#Fitting and printing test accuracy and other measures
model.fit(x,y,epochs=40, batch_size=10)

_,test_acc=model.evaluate(x_v,y_v)

print('Average accuracy:', np.mean(score))
print("Test Accuracy: ", test_acc)
y_pred=model.predict_classes(x_v)

y_pred=np.ma.make_mask(y_pred)

kappa=cohen_kappa_score(y_v,y_pred)
print("Kappa: ", kappa)
print("Recall: ", recall_score(y_v,y_pred))
print("Precision: ", precision_score(y_v,y_pred))
print("confussion: ", confusion_matrix(y_v,y_pred))

#########################
#Predicting Test File
#########################
#Selecting only signigicative features
x_t=x_t.loc[:,selector.get_support()]
#Predicting classes
results=pd.DataFrame()
for i in range(1,11):
    #training neural network
    model=keras.Sequential([
        keras.layers.Input(shape=selector.n_features_.item()),
        keras.layers.Dense(45,activation="relu",kernel_initializer='orthogonal'),
        keras.layers.Dense(5,activation="relu",kernel_initializer='orthogonal'),
        keras.layers.Dense(2, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", \
        metrics=["accuracy"])

    #Fitting and printing test accuracy and other measures
    model.fit(x,y,epochs=40, batch_size=10)
    y_test=model.predict_classes(x_t)
    #appending results to later vote
    results[i]=y_test[:]

#voting criteria across a 10-fold prediction
results["Class"]=np.where(results.sum(axis=1)>=5, True,False)
print(results.head())
#Saving Results
results.to_csv("data\\test_ann.csv", index=False)
print("test results shape: ", results.shape)
