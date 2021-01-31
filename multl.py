# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:42:19 2020

@author: DELL
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as py
import numpy as np

df=pd.read_csv(r"C:\Users\DELL\Documents\multiple.csv")
df.head()
df.describe()

dummy=pd.get_dummies(df['sex'])
dummy1=pd.get_dummies(df['smoker'])
dummy2=pd.get_dummies(df['region'])

df2=pd.concat((df,dummy), axis=1)
df3=pd.concat((df2,dummy1), axis=1)
df4=pd.concat((df3,dummy2), axis=1)

cdf=df4[['age', 'bmi', 'male', 'female', 'children', 'charges']]
cdf.head(1000)

plt.scatter(cdf.bmi, cdf.charges, color='blue')
plt.xlabel("bmi")
plt.ylabel("charges")
plt.show()

msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]

from sklearn import linear_model
regr=linear_model.LinearRegression()
x=np.asanyarray(train[['age', 'bmi', 'male', 'female', 'children']])
y=np.asanyarray(train[['charges']])
regr.fit(x,y)

print('Coefficients: ', regr.coef_)
print('Intercepts: ', regr.intercept_)

#In this 6d plot is difficult so graph wilkl not happen

y_hat=regr.predict(test[['age', 'bmi', 'male', 'female', 'children']])
x=np.asanyarray(test[['age', 'bmi', 'male', 'female', 'children']])
y=np.asanyarray(test[['charges']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))