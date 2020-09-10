#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from pandas import DataFrame as df
from sklearn import tree
from sklearn.model_selection import (KFold, cross_val_score,
                                    GridSearchCV)
from sklearn.feature_selection import RFECV
from scipy import stats
import seaborn as sns
from graphviz import Source
from sklearn.metrics import (confusion_matrix,cohen_kappa_score,recall_score,
                    precision_score)
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

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
est=tree.DecisionTreeClassifier(criterion = "entropy", random_state=21)
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
plt.savefig("plots\\featdt.pdf", bbox_inches='tight')
plt.close()

#####################
#Decision Tree
#####################
###########
#Unpruned
###########
clf = tree.DecisionTreeClassifier(criterion = "entropy", random_state=21)
clf.fit(x_b,y)
y_pred=clf.predict(xv_b)
kappa=cohen_kappa_score(y_v,y_pred)
score_2 = cross_val_score(clf, x_b, y, cv=k_fold, n_jobs=-1)
print('Average accuracy:', np.mean(score_2))
print("Kappa: ", kappa)
print("Score DT: ", clf.score(xv_b,y_v))
print("Depth DT: ", clf.get_depth())
print("Recall: ", recall_score(y_v,y_pred))
print("Precision: ", precision_score(y_v,y_pred))
print("confussion: ", confusion_matrix(y_v,y_pred))

#Decision tree plot
graph = Source(tree.export_graphviz(clf, \
    out_file=None, feature_names=x.loc[:,\
        selector.get_support()].columns))
graph.format = 'pdf'
graph.render('plots\\dt')

#############
#pruned
############
#parameter grid
lf=np.arange(2,30,2)

#Setting up DT and Gridsearch
cflp = tree.DecisionTreeClassifier(criterion = "entropy", random_state=21)
clfp = GridSearchCV(estimator=cflp, param_grid=dict(min_samples_leaf=lf), \
        n_jobs=-1, cv=k_fold)

#Fitting and printing CV accuracy
clfp.fit(x_b,y) #fitting
print("best min:",clfp.best_estimator_.min_samples_leaf)
print("average score:",clfp.best_score_)

#DT plot
graphp = Source(tree.export_graphviz(clfp.best_estimator_,\
   out_file=None, feature_names=x.loc[:,selector.get_support()].columns))
graphp.format = 'pdf'
graphp.render('plots\\dtp')

#Test accuracy and other measures
y_pred=clfp.predict(xv_b)
print("Kappa: ", cohen_kappa_score(y_v,y_pred))
print("Score: ", clfp.score(xv_b,y_v))
print("Recall: ", recall_score(y_v,y_pred))
print("Precision: ", precision_score(y_v,y_pred))
print("confussion: ", confusion_matrix(y_v,y_pred))

#plot of variables (to show separation between classes)
sns.scatterplot(x.iloc[:,18], x.iloc[:,19], hue=y)
plt.savefig("plots\\mixed.pdf", bbox_inches='tight')
plt.close()
