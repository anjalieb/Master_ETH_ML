## Overview
This submission contains 3 Python scripts, which all need to be placed in the same folder, along with the data file 'train.csv'.
The results are obtained by running main.py from the command line. No arguments need to be given if the files are correctly placed.
The final results are generated in the file 'weights.csv'. Intermerdiary results used for deciding which model to use are placed in the file of the same name.

The scripts are:

* ridge_regression.py contains functions necessary to perform k-fold cross-validation with Ridge regression on a given dataset.

* Task1b_auxiliary.py contains different functions necessary to perform task 1.b. 
- it loads the training data
- it defines the feature mapping indicated in the task
- in a third part it splits the dataset between a training and test set
- part 4 explores different regression models by performing 5-fold cross validation on the training subset using a Ridge regression model and different Ridge penalties and calculating their average RMSE
- part 5 then computes the RMSE of each of these models on the test subset.
- This script geenrate a csv file named 'auxiliary_results.csv' which allowed to comapre the different models and choose the one to be implemented. The metric used for comparison is simply the RMSE, both over a 5-fold cross validation on the training subset, and on the test subset.
### Model selection
Using the different reported RMSE, it appears that the best model is a Ridge regression model with a penalty equal to 10, which was then used to generate the results in the file main.py.

* main.py trains the chosen model (a Ridge regression model with a penalty equal to 10, and parameter fit_intercept=False in order to be able to fit the constant feature) over the entire dataset and outputs the resulting weights in the file 'weights.csv'
