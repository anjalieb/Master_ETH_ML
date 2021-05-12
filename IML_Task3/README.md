## Requirements
The same list of requirements as in task2 can be used. See the file requirements.txt
The code can be found in final_submission_task3.py. It can be run as it, provided the training and test data files are in the same folder.

## Task Descripiton
- The goal of this task is to classify mutations of a human antibody protein into active (1) and inactive (0) based on the provided mutation information.
- The mutations differ from each other by 4 amino acids in 4 respective sites. The sites or locations of the mutations are fixed.
- The biological and chemical aspects can be abstracted to solve this task.

## Report
### Feature transformation
We have categorical features, hence a data transformation was defined in which for each position all possible amino acids (20 different ones) are taken and one hot encode the sequence.
Finally, the feature representation that we will use for training consists of 80 features respesenting the 20 amino acids that might be at each position and encoding as 1 if the correct amino acid is given at that position, 0 otherwise.
For prediction, a multi-layer perceptron (MLP) Classifier which uses the log-loss function and stochastic gradient descent.
By using GridSearchCV, the best regularization parameter was deduced because we want to avoid overfitting.
