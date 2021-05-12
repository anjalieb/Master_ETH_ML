

# *This notebook will be the main skeleton for data extraction and running, and will condense the code in a hopefully readable manner*

# In[3]:



## From Python libraries

import pickle

import seaborn as sns

import pandas as pd
pd.set_option('display.max_rows', 20)

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
from sklearn import svm
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor

import time
# define a random seed for all task 
import random
random.seed(5)


# # Part 0: Importing the data 

# In[4]:


# The files are submitted with the code
train_data_file='train_features.csv'
train_labels_file='train_labels.csv'
test_data_file='test_features.csv'

data_train=pd.read_csv(train_data_file)

labels_train=pd.read_csv(train_labels_file)
# index labels_train by pid
labels_train.set_index('pid', drop=True, inplace=True)
    
data_test=pd.read_csv(test_data_file)


# # Part I: Data preprocessing pipeline

# In[5]:


""" organise_data :

Input: a dataframe indexed by pid
Output: a dictionnary indexed by pid, in which all times run from 1 to 12

organise_data simply transforms the original dataframe into a dictionary and sets the time scale from 1 to 12
    """
def organise_data(input_data):
    patients_features={}
    for pid in input_data['pid'].unique():
        #print(pid)
        patient_data=input_data[input_data['pid']==pid]
        index=patient_data.index
        offset=patient_data.at[index[0], 'Time']-1
        for i in (index):
            patient_data.at[i,'Time']-=offset
        patients_features[pid]=patient_data
    return(patients_features)


# In[6]:


""" extract_features :

Input: a dictionary indexed by pid in which all times run from 1 to 12.
Output: a dataframe indexed by pid, in which features have been collapsed into a 1*n dimensional vector. 
This vector has dropped the features Time and pid, and has kept the feature Age. All other features have been expanded
into :
    - a min value
    - a max value
    - a count value (number of non NaN)
    - a median value
    - a mean value
    - a first measurement value
and - a last measurement value.

extract_features does the feature engineering as described in the output.
    """
def extract_features(dic): # dic is a dictionary indexed by pid compiling the raw data from patients in the form of one dataframe per patient.
    #Times have been modified to range from 1 to 12.
    # An example of such dictionary is patients_features_train
    # Added an argument dic to be able to run it on the test features later
    patients_features_engineered={}
    # Loop through all patients
    for (pid, df) in dic.items():
        # Loop through all features
        patients_features_engineered[pid]={}
        for feature in df.columns:
            if feature in ["pid", "Time"]:
                # We don't keep those features
                continue
            elif feature=="Age":
                # We keep the age as it is
                patients_features_engineered[pid]["Age"]=df["Age"].max()
            else:
                # We get min, max, count, median, mean, 1st measurement, last measurement
                current_series=df[feature].dropna()
                patients_features_engineered[pid][feature+"_count"]=len(current_series.index)
                patients_features_engineered[pid][feature+"_max"]=current_series.max()
                patients_features_engineered[pid][feature+"_min"]=current_series.min()
                patients_features_engineered[pid][feature+"_median"]=current_series.median()
                patients_features_engineered[pid][feature+"_mean"]=current_series.mean()
                
                # Obtain the 1st and last measurements
                rowids=current_series.index
                try:
                    first=rowids[0]
                    last=rowids[-1]
                    patients_features_engineered[pid][feature+"_first"]=current_series[first]
                    patients_features_engineered[pid][feature+"_last"]=current_series[last]
                except:
                    print("There are only nans in feature %s of patient %s"%(feature,pid))
                
    # Then we transform all the obtained dictionaries into dataframes and fuse them
    
    # I know pid 1 is in the dataset
    to_concatenate=[]
    for (pid, dic) in patients_features_engineered.items():
        print(pid)
        patient_df=pd.DataFrame.from_records(patients_features_engineered[pid], index=[pid])
        to_concatenate.append(patient_df)
    patients_features_transformed=pd.concat(to_concatenate)

    # Return both the dataframe and the dictionary
    return(patients_features_transformed)
        


# In[7]:


"""
dropping_features

Input: df_training_features: 
    - a dataframe, most likely the output of extract_features when run on the training set
    - percentage, the maximal percentage of NaN accepted for a feature to be kept for prediction
Output: The list of features to keep and use for prediction

dropping_features calculates the list of engineered features which have less than 30% of the values missing
"""

def dropping_features(df_training_features, percentage):
    # create a list
    keep_features = []
    for feature in df_training_features.columns:
        series = df_training_features[feature]
        # take whole column
        total_number = len(series.index)
        # total number of patients
        series_no_na = series.dropna()
        # remove all rows which contain NA
        number_no_na = len(series_no_na.index)
        # number of patients having no NA for this feature
        ratio = number_no_na/total_number
        # if the ratio is < 1-percentage it means that the feature has in more than percentage% of all row-entires NA
        # so the ratio needs to be >= to 1-percentage because we want to drop a feature that has more than percentage% of missing values
        if ratio >= 1-percentage:
            keep_features.append(feature)

    return keep_features


# In[8]:


"""
prepare_features

Input:  engineered_features, the dataframe output of extract_features
        features_to_keep, a list of the feature having sufficiently few NaNs to be deemed useful for prediction
Ouput: a dataframe containing the final data used for prediction 

prepare_features:
 1) Uses the output of dropping_features to remove all features with too many NaNs
 2) Does data imputation on the kept features by replacing NaNs by the median
 3) Standardises the data using the StandardScaler class as this will be needed for the classification step 
"""

def prepare_features(features_to_keep, engineered_features):
    # engineered_features is a dataframe containing all features after feature engineering (min, max, count...)
    # It has all times set from 1 to 12
    
    #1) Keep only the relevant features
    pruned_features=engineered_features[features_to_keep]
    
    #2) Do imputation first
    imputed_features_list=[]
    for feature in pruned_features.columns:
        series=pruned_features[feature]
        series=series.fillna(series.median())
        imputed_features_list.append(series)
    imputed_features=pd.concat(imputed_features_list, axis=1)

    #3) standardise the data
    scaler=StandardScaler()
    standardised_features=scaler.fit_transform(imputed_features)
    standardised_features_df=pd.DataFrame(standardised_features, columns=features_to_keep, index=engineered_features.index)
    return(standardised_features_df)
    
    


# # Part 2: Prediction subtasks
# 
# ## 2.1 : Setting up the training set into training subset, testing subset, and setting up cross validation

# In[9]:


"""
subset_splitting

Input: df_training_final, the final preprocessed training data under the form of a dataframe 
       labels_train, the different labels for the training points
       test_size, the relative size of the test subset (percentage of initial training dataset, default 20%)

Output: df_train: datapoints of training subset
        df_test: datapoints of testing subset
        labels_train: labels of training subset
        labels_test: labels of testing subset
        
subset_splitting splits the training data into a training and a validation subset
"""

def subset_splitting(df_training_final, labels_train, testsize=0.2):
    df_train, df_test, labels_train, labels_test = train_test_split(df_training_final,labels_train,test_size=testsize, random_state=35)
    return(df_train, df_test,labels_train, labels_test)


# ## 2.2: Subtasks 1 and 2: classification

# In[10]:


"""
find_best_model

Input: label, the label we want to predict,
       training_data, the training points (without the validation set) upon which to perform the cross-validation
       labels_train, the labels of the training points

Output: the linear SVM model and the Logistic regression model with the highest auc score on the validation set

find_best_model performs a hyperparameter search for 2 regularised types of classifiers (linear svm and 
logistic regression) using GridSearchCV. Importantly, all GridSearchCV objects are saved into a folder named 
GridSearchResults. Only those saved objects will be usde from now on.

"""
def find_best_model(label, training_data, labels_train): #the label we want to predict
    reference_label=labels_train[label]
    # A linear SVM model
    svm_lin=svm.SVC(probability=True, kernel='linear',class_weight='balanced', random_state=5)
    # A RBF kernel SVM -> forget about it, it doesn't work well
    #svm_rbf=svm.SVC(probability=True, kernel='rbf', class_weight='balanced', random_state=5)
    # A logistic regression model
    log=LogisticRegression(class_weight='balanced',random_state=5, verbose=True)
    
    # For each one run a 5-fold cross_validation and optimise over the hyperparameters
    clf_lin=GridSearchCV(svm_lin, {'C':[0.001,0.01,0.1,1]}, scoring='roc_auc_ovo_weighted', cv=4,verbose=4,
                         return_train_score=False)
    #clf_rbf=GridSearchCV(svm_rbf, {'C':[0.001,0.1,1], 'gamma':[0.001,0.1,1,10,100]},scoring='roc_auc_ovo_weighted', cv=5,verbose=4, return_train_score=False)
    clf_log=GridSearchCV(log,  {'C':[0.00001,0.0001,0.001,0.1,1,10]}, scoring='roc_auc_ovo_weighted', cv=4,verbose=4,
                         return_train_score=False)
    
    # Fit and return the results
    clf_lin.fit(training_data, reference_label)
    clf_log.fit(training_data,reference_label)
    
    # Save the objects
    pickle.dump(clf_lin, open('GridSearchResults/clf_lin_%s'%label, 'wb'))
    pickle.dump(clf_log, open('GridSearchResults/clf_log_%s'%label, 'wb'))
    
    # Return the best logistic model, and the best linear SVM model
    return(clf_log.best_estimator_, clf_lin.best_estimator_)


# In[11]:


"""
train_best_model_and_predict

Input:  label, the label to predict
        model, the model to train, chosen from the models output by find_best_model
        df_train, the points on which to perform the training, 
        labels_train, the labels of the training point
        df_testing_final, the test points upon which to finally predict
Output: model, the fitted model
        prediction_df, a dataframe with the predictions for the label of the test set
"""

def train_best_model_and_predict(label, model, df_train, labels_train, df_testing_final):
    model.fit(df_train, labels_train[label])
    predictions=model.predict_proba(df_testing_final)
    # We want the probability of the positive outcome
    predictions=predictions[:,1]
    # We return a series indexed by pid, label
    prediction_df=pd.DataFrame(predictions, index=df_testing_final.index, columns=[label])
    return(model, prediction_df)
    
    


# ## 2.3: Average value prediction

# **Create and train model for the regression task**

# In[12]:




def create_and_train_model (training_data, training_labels):
    
### I would like to put the part of 'kernel='poly', degree=1' as a parameter,
###but so far i get too many compilation errors
    #The MultiOutputRegressor is being used
    #Parameters are the best parameters to use wth the SVR
    #Training data is the preprocessed data
    #Training_labels are the ground truths from the training data
    clf = MultiOutputRegressor(svm.SVR(kernel='poly',
                               degree=1,
                               verbose=False,
                               cache_size=500),n_jobs=-1)
    start_time = time.time()
    clf.fit(training_data,training_labels)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return clf


# In[13]:


def prepare_dataframe(labels,index,prediction):
    #labels are the labels for the subtask
    #index is the index of the pid from the patients
    #prediction is the prediction data we get from our model
    
    #prepare output dataframe with the correspondant labels index and prediction data
    df_out = pd.DataFrame(columns = labels, index=index)
    
    for i, label in enumerate(labels):
        df_out[label] = prediction[:,i]
    
    return(df_out)

