All the code has initially been written in jupyter notebooks. Those could however not be included due to their size.

## Installation
to be able to run the script, the files included in the task description (test_features.csv, train_features.csv, train_labels.csv) MUST be copied in THE SAME folder ie in Task2/scripts. (we could not include them due to their great size)
Otherwise the code WILL NOT run

However, we have provided the script version in the folder scripts. Those can be run directly from the command line without input arguments by first running code_skeleton.py and then main.py.


Another possibility is to change the relative paths on top of the file code_skeleton.py


The requirements for this code are entirely included in those for the IML class git.  The corresponding requirements.txt file is also provided here.

This code is very long running, and as such include saving objects with the module pickle. The relative paths have already been set correctly so that the objects saved when running the code can be loaded again.



For you, to get an overview of the task2 in the course 'Introduction to Machine Learning' by Prof. Krause, at ETH (8 ECTS):
The idea of the task was to apply ML approaches to real-word problems and to handle data artifacts like missing values, imbalance of labels or heavy-tailed distributions in the data.
The challenge lied in the appropriate pre-processing of the data.
We were given a trainig set including labels and a test set without labels.
The goal was to predict rarely-occuring events such as the prediction of a sepsis of a patient in the Intensive Care Unit (subtask2) or wether a certain medical test is ordered by a clinican during the stay of a patient in the ICU (subtask 3)
In subtask 3, vital signs of the patient state were predicted.
I include here the report I handed in togheter with the code. The following report shortly describes what method were used to solve the prediction task:

1. DATA IMPUTATION
For every patient the time scale is set form 1 to 12.
A dataframe has been created, indexed by pid, in which all features have been collapsed into an 1*n dimensional array.
In this array, all features (except for the feature age) were expanded into a min value, a max value, a count variable (number of NaN), a median value, a mean value, a first measurement.
In order to catch the evolution of the data of a feature, the first and last measurement were included.
Features containing more than 30% NaN were dropped. All other features having sufficiently few NaNs to be deemed useful for prediction are kept.
In these features, the NaNs were replaced by the median.
Tha data was standarised using StandardScaler as this is needed for the Classification step.

2. SUBTASK 1 & 2
The best model was chosen based on its ROC score on the training and test Cross-Validation set. A linear SVM model with different parameters as well as a logistic regression model with different parameters were tested.
For every label individually, the best model with its associated best parameter were selected for the final prediction on the test set.

3. SUBTASK 3
A multioutput regressor model SVR was used to train multiple labels at the same time. It turned out that SVR model with a linear kernel turned out doing acceptable.
