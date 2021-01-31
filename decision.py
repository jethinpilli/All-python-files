# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 23:47:38 2020

@author: DELL
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

my_data=pd.read_csv(r"C:\Users\DELL\Documents\drug.csv")
my_data[0:5]

X=my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

from sklearn import preprocessing
le_sex=preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1]=le_sex.transform(X[:,1])

le_BP=preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2]=le_BP.transform(X[:,2])

le_Chol=preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3]=le_Chol.transform(X[:,3])

X[0:5]

y=my_data[['Drug']]
y[0:5]

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset= train_test_split(X, y, test_size=0.3, random_state=3)
print("Train set: ", X_trainset.shape, y_trainset.shape)
print("Test set: ", X_testset.shape, y_testset.shape)

drugTree=DecisionTreeClassifier(criterion="entropy", max_depth=5)
drugTree

drugTree.fit(X_trainset,y_trainset)
predtree=drugTree.predict(X_testset)

print(predtree[0:5])
print(y_testset[0:5])

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predtree))






 
