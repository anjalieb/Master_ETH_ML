# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:44:49 2021

@author: anjal
"""

""" The goal of this task is to classify mutations of a human antibody protein
into active (1) and inactive (0) based on the provided mutation information. 
Each line in train.csv corresponds to a single mutation. The dataset contains 
112000 rows, where each row is associated with a mutation described by a sequence 
of four letters (amino acids) and its activity (label).
it is very important to detect nearly all active mutations such that they can be evaluated. 
Hence we need to maximize recall (true positive rate), but at the same time we want to 
have equally good precision. Therefore, we use F1 score which captures both precision 
and recall. 
"""
import pandas as pd
import seaborn as sns
import numpy as np
from math import *
import pickle
import matplotlib.pyplot as plt
import random
random.seed(15)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import f1_score

# %% DATA IMPORTATION
train_file = 'train.csv'
data_train = pd.read_csv(train_file)
test_file = 'test.csv'
data_test = pd.read_csv(test_file)

# %% FEATURE TRANSFORMATION

# feature transformation of training data: Initially we planned to use all combinations of Amino Acids (AA) as features, assuming that creation of an active site might result from interaction of non-neighboring Amino Acids. This is what we do here, this feature representation was used for data visualisation
def feature_transformation_train(my_df):
    """ Takes a dataframe with training data as input and applies the wanted feature transformation on it.
        
        Here we have decided to obtain all subsequences of length 1 to 4, assuming that stabilising interactions
        can occur via AA far from each other in the primary sequence, due to the folding of the protein.
    """
    df_list=[]
    for i in my_df.index:
        current_row=my_df.iloc[i]
        sequence=current_row['Sequence']
        activity=current_row['Active']
        
        transformation_dic={'1':sequence[0],'2':sequence[1], '3':sequence[2],'4':sequence[3],
                              '12':sequence[0:2],'13':sequence[0]+sequence[2], '14':sequence[0]+sequence[3],
                              '23':sequence[1]+sequence[2], '24':sequence[1]+sequence[3], '34': sequence[2]+sequence[3],
                              '123':sequence[0]+sequence[1]+sequence[2], '124':sequence[0]+sequence[1]+sequence[3], 
                              '134':sequence[0]+sequence[2]+sequence[3], '234':sequence[1]+sequence[2]+sequence[3], 
                              '1234':sequence, 'Active':activity}
        new_df=pd.DataFrame.from_records(transformation_dic, index=[i])
        df_list.append(new_df)
    return(pd.concat(df_list, axis=0, ignore_index=True))

# feature transformation of test data which is identical to the training data except the column of 'Activity' is left out as our test data is unlabelled
def feature_transformation_test(my_df):
    """ Takes a dataframe with training data as input and applies the wanted feature transformation on it.
        
        Here we have decided to obtain all subsequences of length 1 to 4, assuming that stabilising interactions
        can occur via AA far from each other in the primary sequence, due to the folding of the protein.
    """
    df_list=[]
    for i in my_df.index:
        current_row=my_df.iloc[i]
        sequence=current_row['Sequence']
        
        transformation_dic={'1':sequence[0],'2':sequence[1], '3':sequence[2],'4':sequence[3],
                              '12':sequence[0:2],'13':sequence[0]+sequence[2], '14':sequence[0]+sequence[3],
                              '23':sequence[1]+sequence[2], '24':sequence[1]+sequence[3], '34': sequence[2]+sequence[3],
                              '123':sequence[0]+sequence[1]+sequence[2], '124':sequence[0]+sequence[1]+sequence[3], 
                              '134':sequence[0]+sequence[2]+sequence[3], '234':sequence[1]+sequence[2]+sequence[3], 
                              '1234':sequence}
        new_df=pd.DataFrame.from_records(transformation_dic, index=[i])
        df_list.append(new_df)
    return(pd.concat(df_list, axis=0, ignore_index=True))
        
# transform the data_train calling the function feature_transformation_train
transformed_traindata = feature_transformation_train(data_train)
# transform the data_test calling the function feature_transformation_test
transformed_testdata = feature_transformation_test(data_test)

# We have categorical features, hence we define a new data transformation in which for each
# position we take the 20 AA that are possible and one hot encode the sequence. However, once one-hot encoded, it is not necessary to keep all our previously engineered features since they can be obtained as polynomial functions of the features named '1' ,'2, '3', '4' (respectively AA in the 1st, 2nd, 3rd, 4th position).

# Hence the feature representation that we will use for training consists of 80 features representing the 20 AA we might have at each position and encoding as 1 if it is the correct AA in the given position, 0 otherwise.
df_train=transformed_traindata[['1', '2', '3', '4']]
df_test=transformed_testdata[['1', '2', '3', '4']]
 
# Convert categorical variable into dummy/indicator variables
# df_train_enc and df_test_end have 80 columns: each AA is represented at each position
df_train_enc=pd.get_dummies(df_train) 
df_test_enc=pd.get_dummies(df_test)

# Split the data between a training set and a validation set 75-25
feat_train_enc, feat_valid_enc, activ_train, activ_valid = train_test_split(df_train_enc, transformed_traindata["Active"], test_size=0.25, random_state=5)

# %% PREDICTION

# As visualisation suggested that nonlinear features (namely product of AA at positions3 and 4) could be very discriminant, we chose to train a non linear classifier.

#  Predictions using multi-layer perceptron (MLP) Classifier
# MLP Classifier uses the log-loss function and stochastic gradient descent
# a GridSearchCV with following regularization parameter (0.0, 0.01, 0.001, 0.0001, 1e-05, 1e-06) and scoring = 'f1' was performed


best_mlp_model = MLPClassifier(alpha = 0.0001, verbose=True).fit(df_train_enc,transformed_traindata["Active"])
y_predicted = best_mlp_model.predict(df_test_enc)

# %% Generate Output file

# write the results into csv file
y_predicted_dataframe=pd.DataFrame(y_predicted)
y_predicted_dataframe.to_csv('results.csv', header=False, index=False)

# %%