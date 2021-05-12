## Installation
to be able to run the script, the files test_features.csv, train_features.csv, train_labels.csv MUST be copied in THE SAME folder ie in Task2/scripts.
Otherwise the code WILL NOT run

However, the script version has been provided in the folder scripts. Those can be run directly from the command line without input arguments by first running code_skeleton.py and then main.py.

Another possibility is to change the relative paths on top of the file code_skeleton.py

### Requirements
The requirements for this code are entirely included in requirements.txt file which is also provided here.

This code is very long running, and as such include saving objects with the module pickle. The relative paths have already been set correctly so that the objects saved when running the code can be loaded again.

## Taskdescription
This is not the official, complete task description we got from ETH but a shorted version I created for you.
This task is number 2 out of 5 in the course 'Introduction to Machine Learning' by Prof. Krause, at ETH (8 ECTS).
Task 2 consists of 3 subtasks and the idea was to apply ML approaches to real-word problems and to handle data artifacts like missing values, imbalance of labels or heavy-tailed distributions in the data.
The challenge lied in the appropriate pre-processing of the data.
We were given a training set including labels and a test set without labels. The training and test set and consisted of lots of different medical data for many patients, where each patient was monitored over 12 hours.
The goal was to predict rarely-occuring events such as the prediction of sepsis of a patient in the Intensive Care Unit (subtask2) or wether a certain medical test is ordered by a clinican during the stay of a patient in the ICU (subtask 2)
In subtask 3, vital signs of the patient state were predicted.
I include here the report I handed in togheter with the code. The following report shortly describes what method were used to solve the prediction task.

## Report
### 1. DATA IMPUTATION
A dataframe has been created, indexed by the patient id, in which all features have been collapsed into an 1*n dimensional array.
In this array, all features (except for the feature age) were expanded into a min value, a max value, a count variable (number of NaN), a median value, a mean value and a first measurement.
In order to catch the evolution of the data of a feature, the first and last measurement were included.
Features containing more than 30% NaN were dropped. All other features having sufficiently few NaNs to be deemed useful for prediction are kept.
In these features, the NaNs were replaced by the median.
Tha data was standarised using StandardScaler as this is needed for the Classification step.

### 2. SUBTASK 1 & 2
The best model was chosen based on its ROC score on the training and test Cross-Validation set. A linear SVM model with different parameters as well as a logistic regression model with different parameters were tested.
For every label individually, the best model with its best associated parameter were selected for the final prediction on the test set. GridSearchCV was used to deduce the best model including the best parameters.

### 3. SUBTASK 3
A multioutput regressor model SVR was used to train multiple labels at the same time. It turned out that SVR model with a linear kernel turned out doing acceptable.
