# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 08:49:17 2021

@author: Constance LE GAC
"""

# %% README

#This file should be run from the command line or an IDE. It needs to be in the same folder as the data file 'train.csv', the script ridge_regression.py, and the script Task1b_auxiliary.py

# %% IMPORTATIONS

## Other auxiliary files

import ridge_regression as ridge
import Task1b_auxiliary as aux

## From Python libraries
import pandas as pd
import numpy as np



# Randomness
import random
# set the random seed
random.seed(5)


# %%  LOAD THE DATA


# The training file should be named 'train.csv' and in the same folder as the script
train_file='train.csv'

# Will perhaps have to go back for precision --> using strings?
data_train=pd.read_csv(train_file)
X=np.array(data_train.iloc[:,2:7])
y=np.array(data_train['y'])
Id=np.array(data_train['Id'])


# %% MAIN FUNCTION

## This function trains the model we have chosen (Ridge regression with penalty of 10) on the entire available dataset. It first computes the feature mapping of this set, then trains a Ridge predictor with penalty 10, and finally writes the weights into an csv file named 'weights.csv' which can be found in the same folder as the one the script is run from

if __name__ == '__main__':
    phi=aux.create_feature_map(X)
    ridge_predictor=ridge.Ridge(10, fit_intercept=False)
    ridge_predictor.fit(phi,y, )
    weights=ridge_predictor.coef_
    weights_df=pd.DataFrame(weights)
    weights_df.to_csv('weights.csv', index=None, header=None)