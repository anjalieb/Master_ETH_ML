# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 09:23:41 2021

@author: Constance LE GAC
"""

# %% DESCRIPTION

# This script contains different functions necessary to perform task 1.b. In a first part it loads the training data; in a second part it defines the feature mapping indicated in the task; in a third part it splits the dataset between a training and test set; part 4 explores different regression models by performing 5-fold cross validation on the training subset using a Ridge regression model and different Ridge penalties and calculating their average RMSE, and part 5 then computes the RMSE of each of these models on the test subset.

# Using the different reported RMSE, it appears that the best model is a Ridge regression model with a penalty equal to 10, which we then use to generate our results in the file main.py.

# This file needs to be in the same folder as the data file 'train.csv', the script main.py, and the script ridge_regression.py

# %% IMPORTATIONS


## Other auxiliary files

import ridge_regression as ridge

## From Python libraries

# Data Management
import pandas as pd
pd.set_option('display.max_rows', 10)
import numpy as np

# Randomness
import random
# set the random seed
random.seed(5)

# Mathematical functions
from math import exp
from math import cos

# ML
from sklearn.model_selection import train_test_split

# %% PART 1: LOAD THE DATA


# The training file should be named 'train.csv' and in the same folder as the script
train_file='train.csv'

data_train=pd.read_csv(train_file)
X=np.array(data_train.iloc[:,2:7])
y=np.array(data_train['y'])
Id=np.array(data_train['Id'])

# %%  PART 2: FEATURE MAPPING


# A simple squaring function in order to be able to vecotrise it
def square_func(x):
    return(x**2)


# Given a datapoint, feature_mapping(x) returns phi(x) the vector [phi1(x), phi2(x)...phi21(x)]

def feature_mapping_bis(X):
    square=np.vectorize(square_func)(X)
    exponential=np.vectorize(exp)(X)
    cosine=np.vectorize(cos)(X)
    return(np.concatenate((X, square, exponential, cosine, np.array([1]))))

# create_feature_map creates the feature mapping of a set of datapoints
def create_feature_map(X):
    # X is a set of datapoints
    Phi=[]
    for x in X:
        Phi.append(feature_mapping_bis(x))
    return(np.array(Phi))

# %% PART 3: DEFINE TRAINING AND TEST SET

# As we only have a training set, we have to divide it between training and test set to be able to perform a meaningful training. Let's take a random division of 85% train 15% test (Because I'm again going to use k_fold cross_validation so I want something acceptable. We'll use 5-fold cross-validation to still have a reasonable training and test size.)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=35)

# Create the feature mappings on which we'll work
phi_train=create_feature_map(X_train)
phi_test=create_feature_map(X_test)

# %% PART 4: REGRESSION 

# We have seen in the lecture that linear classifiers can perform fairly well on when associated to a non-linear feature map. I suggest we start with our vanilla linear regression.

# Using Task 1.a as an inspiration, let us also train a Ridge regression with different penalties and see if one of them performs better 1) on average over the 5-fold cross-validation on the training dataset and 2) on the 15% test set we have kept.

def ridge_training():
    results=[]
    regularisation_params=[0,0.001,0.01,0.1,1,10,100]
    for lam in regularisation_params:
        results.append(ridge.CV(5,lam, phi_train, y_train, fit_intercept_bool=False))
    
    # Write the results in an output file
    results_df=pd.DataFrame(np.array(results), columns=['average RMSE over 5-fold CV '], index=['0', '0.001', '0.01', '0.1', '1', '10', '100'])
    return(results_df)
    #results_df.to_csv('ridge_results.csv', index=False, header=False)

# Creates a dataframe which contains the average RMSE over a 5-fold cross-validation of the training subset for a ridge penalty in [0,0.001,0.01,1,10,100]
CV_RMSE=ridge_training()
print(CV_RMSE)

# From these first results, it seems that the use of a penalty does not change much the predition, since the RMSE is within 0.1 of 2 in all cases. It seems a Ridge penalty of 10 yields the best results. Let us now evaluate the prediction of each model over our 15% test set 

# %% PART 5: EVALUATION

# Let us now evaluate the prediction of each model over our 15% test set and compare it to our average RMSE
# Outputs all intermediary results in a file named 'intermediary_results.csv'

def evaluation_on_test_subset():
    results=[]
    regularisation_params=[0,0.001,0.01,0.1,1,10,100]
    for lam in regularisation_params:
        prediction=ridge.perform_Ridge_regression(lam,phi_train,y_train,phi_test, fit_intercept_bool=False)
        results.append(ridge.RMSE_Ridge_regression(prediction,y_test))
    # Compare to results of cross-validation
    RMSE=ridge_training()
    RMSE['RMSE over test subset']=results
    RMSE.to_csv('intermediary_results.csv')
    return(RMSE)

# Print the intermediary results
print(evaluation_on_test_subset())

# For some reason the RMSE is much smaller on the test set... Could that just be an artifact of the split?

# From the information available in intermediar_results.csv, a Ridge regression penalty of 10 seems to be the best compromise, with lowest RMSE on the 5-fold cross-validation, and lowest RMSE on the test set.

# It is thus the model we will use to predict the weights and the submission (see in main.py)

# Nota Bene: Before I tried fitting a model with fit_intercept=True (default for Ridge regression) -> This does not work because we have a constant feature, so we want to ignore the intercept to then be able to put a weight on this feature 
