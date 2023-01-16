# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 19:40:51 2018

@author: natnaelk
"""

#Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the dataset 
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4].values

#Encoding the categorical indepndent variable, creating dummies

#But the first step here we will create three dummy variables, for the three diff states
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#we need to use labelencoder first becasue this changes text to numbers, we cant just jump straight into onehotencoder

onehotencoder = OneHotEncoder(categorical_features = [3])
#The three is the index of the column you want to encode
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
#The library we're using here avoids the dummy variable trap already, so we dont have to reduce 3 dummies to 2
#But other packages you use wont do this automatically so look at code below
X = X[:, 1:]#this code just removed the first dummy variable column in the x matrix

#good to remove column in general best practice and avoid trap regardless of library

#Now when the one column is removed, the model can infer the other state b/c the other two variables will be 0 values
#so the one state will flow into the constant, or default value


#Splitting the Dataset into the Training set and Test Set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)

#Feature Scaling

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) """


# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)



#Building the optimal model using Backward Elimination
#Above we simply used all the variables we started with, but some of these might not be significant etc
#So now were gonna use backward elimination to try to create a more accurate model

import statsmodels.formula.api as sm

#the statsmodels library deosnt take into account that the constant b0 has a coefficient of 1
#most other librarys already account for this, statsmodels doesnt so we need to add a column of 1's

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
#code above were adding the x martrix to this array of 1's, thats why we put it second
#after array code, astype(int) specifices integer so we dont get data error, axis =0 is for row and axis =1 is for column
#np.append appens arrays to matrixes etc and np.ones adds a row or column of ones

X_opt = X[:, [0, 1, 2, 3, 4, 5] ]  #x_opt is the optimal x matrix we want at the end after backward elimination

#So now we gotta creat a new regressor for the stats.model library
#we have new class statsmodel, not using sklearn anymore so we have to create a new object/new regressor

regressor_OLS= sm.OLS(endog = y, exog = X_opt).fit()
#endog is dependent variable #exog is ind variables, also states in the help tab that independent variable isn't included in the intercept, thats y we needed to add array of 1's.

regressor_OLS.summary()
# We look at p values, X2 is the highest so we want to remove it and re fit based off step 3 and 4 of backward elimination

#Re-fit without x2
X_opt = X[:, [0, 1, 3, 4, 5] ]
regressor_OLS= sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# now x1 is highest s o we want to remove it

#Re-fit without x1
X_opt = X[:, [0, 3, 4, 5] ]
regressor_OLS= sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Refit again
X_opt = X[:, [0, 3, 5] ]
regressor_OLS= sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Final Refit
X_opt = X[:, [0, 3] ]
regressor_OLS= sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


#Its a little tricky comparing the index number to the x1,x2, etc from the summary table be dilligent and make sure you're taking out the same variables.
#Easiest way is to notice the order of the coefficents in the summary table and compare them to the order of 
#indexs you have in your table. Or find a way to retitle these variables

