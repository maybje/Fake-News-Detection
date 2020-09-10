#!/usr/bin/env python3
import os.path
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from pandas import DataFrame as df
from sklearn import tree
from scipy import stats
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer,SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from dython import nominal

#Set wd
os.chdir("D:\\Documentos\\Essex\\Machine Learning\\assignment")
cwd=os.getcwd()
#read data
data=pd.read_csv("data\\training_set.csv", header=0)
data2=pd.read_csv("data\\validation_set.csv", header=0)
data3=pd.read_csv("CE802_Ass_2019_Test.csv", header=0)
#print(data.head())

#create feature matrix and feature vectors
#Training set
y=data.iloc[:,-1]
x=data.iloc[:,:-1]
names=list(x.columns)
print("Shape X matrix: ", x.shape)

#validation set
y_v=data2.iloc[:,-1]
x_v=data2.iloc[:,:-1]
print("Shape X_v matrix: ", x_v.shape)

#test set
y_t=data3.iloc[:,-1]
x_t=data3.iloc[:,:-1]
print("Shape X_t matrix: ", x_t.shape)

#standard normalizer
scaler=StandardScaler()
scaler.fit(x)
x_st=scaler.transform(x)

###scaling validation set using distribution of training
x_v=pd.DataFrame(scaler.transform(x_v), columns=names)

###scaling test set using distribution of training
x_t=pd.DataFrame(scaler.transform(x_t), columns=names)

################
##KNN imputation
################
#setting imputers
k_im=KNNImputer(n_neighbors=5, weights="distance")
m_im=SimpleImputer(strategy='median')
dt_im=IterativeImputer(estimator=tree.DecisionTreeRegressor(random_state=21),random_state=21)

#Perform imputation on training set
x_ki=pd.DataFrame(k_im.fit_transform(x_st), columns=names)
x_mi=pd.DataFrame(m_im.fit_transform(x_st), columns=names)
x_dti=pd.DataFrame(dt_im.fit_transform(x_st), columns=names)
x_st=pd.DataFrame(x_st, columns=names)

#density plots for F20 before and after imputation
before=sns.kdeplot(x_st['F20'].notnull(),  bw=0.4, label="Original")
me=sns.kdeplot(x_mi['F20'],  bw=0.4, label="Median")
kn=sns.kdeplot(x_ki['F20'],  bw=0.4, label="5-NN")
dt=sns.kdeplot(x_dti['F20'],  bw=0.4, label="Decission Tree")
plt.xlim(-3, 3)
plt.savefig("plots\\f20.pdf", bbox_inches='tight')
plt.close()

#Exporting latex table of descriptive statistics
print(pd.concat([x_st['F20'].describe(),pd.concat([x_ki['F20'].\
    describe(),pd.concat([x_mi['F20'].describe(),x_dti['F20'].\
    describe()],axis=1)],axis=1)],axis=1).to_latex(float_format="%.2f"))

#saving imputed training set
imputed=x_dti.join(y)
imputed.to_csv("data\\train_imp.csv", index=False)
print("Imputed shape: ", imputed.shape)

##imputing validation_set
x_v=pd.DataFrame(dt_im.transform(x_v), columns=names)

#saving imputed validation set
validation=x_v.join(y_v)
validation.to_csv("data\\val_imp.csv", index=False)
print("Val shape: ", validation.shape)

##imputing test set
x_t=pd.DataFrame(dt_im.transform(x_t), columns=names)
x_levels=scaler.inverse_transform(x_t)

#saving imputed test set
test=x_t.join(y_t)
test.to_csv("data\\test_imp.csv", index=False)
print("test shape: ", test.shape)

test_levels=pd.DataFrame(x_levels).join(y_t)
test_levels.to_csv("data\\test_levels.csv", index=False)
print("test levels shape: ", test_levels.shape)

##################
#Descriptive Plots
##################
#Boxplots
plt.figure(figsize=(10, 4))
sns.boxplot(x="variable", y="value", hue="Class",\
    data=pd.melt(imputed.iloc[:,2:],id_vars=['Class']),palette="Set2")
plt.legend(bbox_to_anchor=(0.045,0.81))

plt.savefig("plots\\boxplot.pdf", bbox_inches='tight')
plt.close()

###################
#Analyzing features
###################
##exporting F1 distribution to latex
print(pd.crosstab(imputed.Class, imputed.F1, normalize='all').to_latex())
chi_s=stats.chi2_contingency(pd.crosstab(imputed.Class, imputed.F1, normalize='all'),\
correction=False)
print("chi 1: ", chi_s) #chi squared test

##xporting F2 distribution to latex
print(pd.crosstab(imputed.Class, imputed.F2, normalize='all').to_latex())
chi_s=stats.chi2_contingency(pd.crosstab(imputed.Class, imputed.F2, normalize='all'),\
correction=False)
print("chi 2: ", chi_s) #chi squared test

#Printing shapes for control
print(imputed.iloc[:,1].shape, imputed.iloc[:,-1:].shape)


#Barplot of categorical variables F1 and F2
sns.countplot(x="variable", hue="Class", \
    data=pd.melt(imputed.iloc[:,[0,1,-1]],id_vars=['Class'])\
    ,palette="Set2")
sns.countplot(x="variable", hue="Class", \
    data=pd.melt(imputed.iloc[:,[0,1,-1]],id_vars=['Class'])\
    ,palette="Set2")
plt.savefig("plots\\f1f2.pdf", bbox_inches='tight')
plt.close()


########################
#Asociation matrix
########################
#Correlation matrix for nominal and categorical variables
nominal.associations(imputed, nominal_columns=['F1','F2',"Class"])
