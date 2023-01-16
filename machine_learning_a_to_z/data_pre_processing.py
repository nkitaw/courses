#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 20:03:30 2018

@author: natnaelk
"""

# Data Preprocessing

#Importing the libraries
import numpy as np # as np syntax is an abrev to 
# use later in your code so you dont have to write import numpy everytime
import matplotlib.pyplot as plt #matplotlib allows you to graph in python
import pandas as pd #used for importing and managing datasets

# Importing the dataset 
dataset = pd.read_csv('data.csv')

#You can use code below to figure out what folder python is linked to
import os 
os.getcwd()

#Creating matrix of features, basically making a matrix for all three 
#idependent variables and another matrix for the dependent variable

X = dataset.iloc[:, :-1].values
# ":" means we take all the values. To the left of comma we are looking at rows
#so we take all of the rows. To the right of the comma we are looking at columns
# so we take all columns besides the last one "-1"
#[rows, columns]<-- basic syntax

#creating dependent variable vector
y = dataset.iloc[:,3].values
#3 because first column is 0 then 1, 2, and finally last column is 3


#Object oriented Summary

#Flow from bigger to more specific is Classes -> Object ->Method

#Class is model of something we want to build
#Object is an instance of the class
#A method is a tool we can use on the object to complete a specific action
    # A method can also be seen as a function that is applied to that object
    #takes some inputs (that were defined in the class) and returns some output
    




#Handling Missing Data
#1. One Method is to take the mean of the values in the columns that contain 
#missing data
    
#were going to use library sci kit learn pre processing
#and from this library were going to import imputer class
    
from sklearn.preprocessing import Imputer

#We imported class Imputer now we need to define object of this class
#stupid and cofunsing but uppercase Imputer is class and lowercase imputer
#is the object of that class

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

#Now we need to fit this imputer object to our matrix of features x
#And were only gonna fit the object to where this is missing data not whole 
#matrix X
imputer = imputer.fit(X[:, 1:3])
#So what remember to take columns 2 and 3 remember python indexes start at 0
#so we start with 1, but also the upper bound isnt included which is why the 3
#is there, so this says give me index 1 and 2  


#So now we just need to replace missing data of matrix x by mean of column
#applying method of imputer object transform, transform replaces missing data
#with mean of the column
X[:, 1:3] = imputer.transform(X[:, 1:3])




#Handling categorical data

#we have to code country and purchase into numbers

#Import library skl.preprocessing library and class labelEncoder and 
#hotlabelencoder because we need to create dummy variables. We dont want to 
#just create variable 1, 2, 3, 4 etc cause then the model will think Germany
#is four times greater than France. They should all be equal so we have to 
#use dummy variable class hotencoder which makes dummy variables


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#labelencoder_X is an object of LabelEncoder, setting object of class
labelencoder_x = LabelEncoder()

#setting column with contries equal to object labelencoder_x fitting to x matrix
X[:, 0] = labelencoder_x.fit_transform(X[:, 0])

#setting object onehotencoder of class, its equal 0 because index is 0, first column
#is country column
onehotencoder = OneHotEncoder(categorical_features = [0])

#fitting object to matrix x
X = onehotencoder.fit_transform(X).toarray()
# after the X theres no :, 0 because we specified earlier index 0 only 1 column
#thats possible

#for y output variable it can be ordinal so we only use label encoder
labelencoder_y = LabelEncoder()

#setting column with contries equal to object labelencoder_x fitting to x matrix
y = labelencoder_y.fit_transform(y)




#Splitting the Dataset into the Training set and Test Set

#Concepts: We create a training dataset so that the machine can learn the coorelation 
#between the inputs and outputs, and then you check to see with the test dataset
#if you can predict correctly with known values. Thats y u split it up.
#Something else can happen if you over train the training dataset you can have 
#overfitting of your model, this can be remedied with regularization techniques
#we dont deal with overfitting and regularization in this lesson, later on

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)
#test size is generally 20 to 30 percent of entire dataset, you want bulk of 
#data to be used learning the model, then a wee bit of it testing the model





#Feature Scaling
#Concept: We do scaling because we want our input variables to be on the same scale
#age is from 1 to 100 salary is 0 to 100,000. lots of models are based of euclidean
#distance, the old school distance formula. We want to make sure both variables
#are on the same scale. This is to prevent one input variable unfairly dominating its
#effect on the dependent variable. So we can standardize or normalize these values
#these two are diff!!

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #for the training set we have to fit & transform
X_test = sc_X.transform(X_test) #for test set we only transform which makes sense
# because we dont need to fit sc_X object to the test set because its already
#fitted to the training set

#Question for later
#could i just write it all in one line instead of setting xc object first
# Anser from evan: better practice to set object first in the beginning
#also you dont have to write standscaler over and over
