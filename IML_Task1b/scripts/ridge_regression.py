# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:06:25 2021

@author: Constance LE GAC
"""

# %% DESCRIPTION

# This script contains functions necessary to perform k-fold cross-validation with Ridge regression on a given dataset, similar to what has been done in Task 1.a.

# This file needs to be in the same folder as the data file 'train.csv', the script main.py, and the script Task1b_auxiliary.py
# %% IMPORTATIONS

# Data Management
import numpy as np

# ML functions
# for the Ridge regression
from sklearn.linear_model import Ridge 
# For splitting the train in set to enable the 10-fold cross-validation
from sklearn.model_selection import KFold 

# Randomness
import random
# set the random seed
random.seed(35)

# Mathematical functions
from math import sqrt

# %% Ridge regression training

#This function performs the k-fold cross validation for a given lambda (ridge regression penalty) and returns the average RMSE over this fold on the dataset X_train with associated labels Y_train.

def CV(k,lam, X_train, Y_train,fit_intercept_bool=True,):
    # lam is the Ridge regularisation parameter, k the parameter for cross-validation, fit_intercept_bool is a boolean indicating whether or not we want to fit an intercept on the Ridge regression (default True, as in the sklearn function)
    
    # Define the K-folds
    
    folds=KFold(n_splits=k) # Default with no shuffling
    # Split X in the different folds
    fold_sets=folds.split(X_train)
    
    # Remember the RMSE for each fold
    RMSE=[]
    for f in fold_sets:
        
        # For each fold define the training set and the test set
        train_fold=X_train[f[0]]
        train_fold_labels=Y_train[f[0]]
        
        test_fold=X_train[f[1]]
        test_fold_labels=Y_train[f[1]]
        
        # For each fold, train a Ridge regression over the training set, and predict over the test set
        ridge_prediction= perform_Ridge_regression(lam,train_fold, train_fold_labels, test_fold, fit_intercept_bool)
        # Compute the RMSE for this fold
        RMSE.append(RMSE_Ridge_regression(ridge_prediction,test_fold_labels))
        
    # Return the mean RMSE over the 10 fold for the given Ridge regularisation parameter    
    return(np.mean(np.array(RMSE)))
    

## This function trains the Ridge regression with parameter lam on a train set train_set using its labels train_set_labels 
## and returns the prediction of this predictor on a test set test_set
def perform_Ridge_regression(lam,train_set, train_set_labels, test_set, fit_intercept_bool=True):
    
    # Trains the Ridge regression model on the train_set, fit_intercept is to set the parameter of the same name in ridge_object.fit (default True as in the sklearn function)
    ridge_object=Ridge(lam,fit_intercept=fit_intercept_bool)
    ridge_object.fit(train_set, train_set_labels)
    
    # Return the predictions of the Ridge regression model on test_set
    return(ridge_object.predict(test_set))


## This function calculates the RMSE between a prediction prediction and the actual label true_labels
def RMSE_Ridge_regression(prediction, true_labels):
    n=len(true_labels)
    sum=0
    for i in range(n):
        sum+= (true_labels[i]-prediction[i])**2
    return(sqrt(sum/n))
