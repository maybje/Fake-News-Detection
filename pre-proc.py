#!/usr/bin/env python3
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
from pandas import DataFrame as df
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split


#Set wd
os.chdir("D:\\Documentos\\Essex\\Machine Learning\\assignment")
cwd=os.getcwd()
#read data
data=pd.read_csv("CE802_Ass_2019_Data.csv", header=0)
#print(data.head())

#create feature matrix and feature vectors
y=data.iloc[:,-1]
x=data.iloc[:,:-1]
names=list(x.columns)

#Class proportions
print(y.value_counts()/y.shape)

#Split Training and Validation test:90% train and 10% test
train_s=0.9
x_t, x_v, y_t, y_v=train_test_split(x,y, train_size=train_s, random_state=21)

#Join features and targets
training=x_t.join(y_t)
validation=x_v.join(y_v)

#Print class proportions of subsets
print(y_t.value_counts()/y_t.shape)
print(y_v.value_counts()/y_v.shape)

#Saving subsets and printing shape for control
training.to_csv("data\\training_set.csv", index=False)
validation.to_csv("data\\validation_set.csv", index=False)
print("Training shape: ", training.shape)
print("Validation shape: ", validation.shape)

#######################
#Exploring Missing Data
#######################
#complete case vector
y_com=y[pd.notna(x.iloc[:,-1])]

#Barplot of original data vs complete cases to see if different
graph_df=y.value_counts(normalize=True).rename("y").to_frame().\
        join(y_com.value_counts(normalize=True).rename("y_com").to_frame())
graph_df.plot(kind='bar', rot=0).legend(["Original", "Complete Case"])
plt.savefig("plots\\missing.pdf", bbox_inches='tight')
plt.close()
#Chi square to test if they are different (MAR)
chi_s=stats.chi2_contingency(np.array([y.value_counts(),y_com.value_counts()]),\
correction=False)
print(np.array([y.value_counts(),y_com.value_counts()]))
print("chi: ", chi_s)

#describe the dataset
print(x.describe().transpose().to_latex(float_format="%.2f"))

##Matrix plot of missing valuse
y_s=y.sort_values()
x_s=x.reindex(y_s.index.tolist())
x_s=x_s.reset_index(drop=True)
fig = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1])
ax1=plt.subplot(gs[0])
sns.heatmap(x_s.isnull(), ax=ax1,cbar=False)
ax2=plt.subplot(gs[1])
cmap = sns.color_palette("coolwarm_r", 2)
sns.heatmap(y_s.to_frame(), ax=ax2,  cmap=cmap)
colorbar=ax2.collections[0].colorbar
r=colorbar.vmax-colorbar.vmin
colorbar.set_ticks([colorbar.vmin + r /2 * (0.5 + i) for i in range(2)])
colorbar.set_ticklabels(list(["False","True"]))
colorbar.ax.tick_params(labelsize=6)
ax1.tick_params(labelsize=8)
ax2.tick_params(axis="x", labelsize=8)
ax2.set_yticklabels([])
ax2.set_yticks([])
for tick in ax2.get_xticklabels():
    tick.set_rotation(90)
plt.tight_layout()
plt.savefig("plots\\matrix_missing.pdf", bbox_inches='tight')
plt.close()
