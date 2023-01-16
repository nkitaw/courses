# -*- coding: utf-8 -*-



#Simple Linear Regression
#Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the dataset 
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # using : -1 syntax is useful if you want mutliple
#x columns it gives you everything besides the last column etc, this is also
#important because it creates a matrix (30,1)
#but for here only one x column you can just do , 0
y = dataset.iloc[:,1].values 

#this is 1 because index starts at 0 
#we want the second column

#Splitting the Dataset into the Training set and Test Set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, 
                                                    random_state = 0)

#looks like this split does mix up observations so its not all of small years
#of observations its evenly distribued the x's with small and large x values

#Feature Scaling

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


#Fitting Simple Linear Regression Model to the Training set"
#basically where the machine is learning the data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #creates object regressor to Linear Regression class
regressor.fit(X_train,y_train)

#Predicting the Test set results

#we will create a vector of predicted values (y_pred)
# *This is very important* ->so regressor object learned the data and now were gonna 
#use regressor object to predict using x_test
#so here instead of using fit, we dont have data we want to fit we already used
#data to learn and (fit) now we want to take this knowledge and predict new values
#based off regressor object we already created and then fitted
y_pred = regressor.predict(X_test) # so here we are using x test to create
# y_pred tosee what the model thought the y should be compared to our y_test we have from the beg

#So now we can compare y_pred to y_test and see how close we are!



#Visualising the Training set data
#The goal here is to plot the training set values, then to plot the training set
#regression line
plt.scatter(X_train, y_train, color = 'red',) #this line simply plots training data points
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #this plots the regression line for training dataset
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

#Visualising the Test set results

plt.scatter(X_test, y_test, color = 'red',) #this line simply plots test set data points
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #this plots the regression line with object regressor 
#we used to fit to training set, this regression line ISNT a regression line fitted of of X_test. Thats why this regression line is the same one as before. code is same as up top
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
