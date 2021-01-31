# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:14:57 2020

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as py
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

cell_df=pd.read_csv(r"C:\Users\DELL\Documents\cell.csv")
cell_df.head()

ax=cell_df[cell_df['Class']==4][0:50].plot(kind='scatter',x='Clump',y='UnifSize', color='Darkblue', label='malignant');
cell_df[cell_df['Class']==2][0:50].plot(kind='scatter',x='Clump',y='UnifSize',color='Yellow',label='benign', ax=ax);
plt.show()

cell_df.dtypes

cell_df=cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc']=cell_df['BareNuc'].astype('int')
cell_df.dtypes

X=np.asanyarray(cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']])
X[0:5]

cell_df['Class']=cell_df['Class'].astype(int)
y=np.asarray(cell_df[['Class']])
y[0:5]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=4)
print("Train Set: ", X_train.shape, y_train.shape)
print("Test size: ", X_test.shape, y_test.shape)

from sklearn import svm
clf=svm.SVC(kernel='poly')
clf.fit(X_train, y_train)

yhat=clf.predict(X_test)
yhat[0:5]

from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted')

#from sklearn.metrics import jaccard_similarity_score
#jaccard_similarity_score(y_test, yhat)