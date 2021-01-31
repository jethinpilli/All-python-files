# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:40:05 2020

@author: DELL
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as py
import numpy as np

df=pd.read_csv(r"C:\Users\DELL\Documents\simple.csv")
df.head()
df.describe()

viz=df[['SAT', 'GPA']]
viz.hist()
plt.show()

plt.scatter(df.SAT, df.GPA, color='maroon')
plt.xlabel("SAT")
plt.ylabel("GPA")
plt.show()

msk=np.random.rand(len(df))<0.8
train=df[msk]
test=df[~msk]

from sklearn import linear_model
regr=linear_model.LinearRegression()
train_x=np.asanyarray(train[['SAT']])
train_y=np.asanyarray(train[['GPA']])
regr.fit(train_x, train_y)

print("Coefficients : ", regr.coef_)
print("Intercepts : ", regr.intercept_)

plt.scatter(train.SAT, train.GPA, color= 'violet')
py.plot(train_x, regr.coef_[0][0]*train_x+regr.intercept_, '-r')
plt.xlabel("SAT")
plt.ylabel("GPA")

plt.scatter(test.SAT, test.GPA, color='black')
plt.xlabel("SAT")
plt.ylabel("GPA")
py.plot()

from sklearn.metrics import r2_score
test_x=np.asanyarray(test[['SAT']])
test_y=np.asanyarray(test[['GPA']])
test_y_=regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


