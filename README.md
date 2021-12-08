# Selection of ML Coding experiences 
Here is a selection of Homework assignements and Research Projects during my Master studies at ETH.

## Semester Research Project at Prof. Karsten Borgwardt's lab at D-BSSE, ETH Zurich
During 6 weeks I investigated in the Predictiion of 1p/19q co-deletion status of low-grade glioma (LGG) from magnetic resonance (MR) images using Convolutional Neural Network.
What my project was about:
Patients having Low-grade gliomas (LGG) with the 1p/19q co-deletion mutation have been shown to better respond to treatment than patients without the mutation translating to improved survival prognosis, too. However, current methods to detect this mutation are brain-tissue biopsy or surgical resection of the tumor which are highly invasive.
The aim of this lab rotation is to use a simple convolutional neural network (CNN) to predict the 1p/19q status from magnetic resonance (MR) images as a non-invasive method. The data comprised a total of 112 LGG patients with biopsy-proven 1p/19q status. CNN hyperparameters were optimized by grid search leading to a 5-fold cross-validation best mean(standard deviation) test set performance of 67.2% (7.3%) accuracy, 71.7% (7.6%) AUCroc and 76.3% (5.8%) average precision score. Hence, the model is not yet applicable for clinical use but may require further investigation of hyper-parameter tuning, inclusion of different contrast images as well as data augmentation to boost performance. Once the presented method is improved it could potentially be used as an alternative to surgical biopsy for predicting 1p/19q co-deletion status.

## Introduction to Machine Learning (IML) course 
This course gives a wide overview into Machine Learning techniques:
- Linear regression (overfitting, cross-validation/bootstrap, model selection, regularization, [stochastic] gradient descent)
- Linear classification: Logistic regression (feature selection, sparsity, multi-class)
- Kernels and the kernel trick (Properties of kernels; applications to linear and logistic regression); k-nearest neighbor
- Neural networks (backpropagation, regularization, convolutional neural networks)
- Unsupervised learning (k-means, PCA, neural network autoencoders)
- The statistical perspective (regularization as prior; loss as likelihood; learning as MAP inference)
- Statistical decision theory (decision making based on statistical models and utility functions)
- Discriminative vs. generative modeling (benefits and challenges in modeling joint vy. conditional distributions)
- Bayes' classifiers (Naive Bayes, Gaussian Bayes; MLE)
- Bayesian approaches to unsupervised learning (Gaussian mixtures, EM)

### Projects in IML
The Project work in IML is a team work of 2 students. I collaborated with my fellow student for the following tasks.
- Task 1b
- Task 2
- Task 3

## Data Mining 2 (DM) course
This course presents advanced topics in data mining and its applications, it is about finding patterns and statistical dependencies in large databases, to gain an understanding of the underlying system from which the data were obtained.

- Dimensionality Reduction
- Association Rule Mining
- Text Mining
- Graph Mining

### Projects in DM2
The project work in DM2 is an individual work
- DM2_PCA_SVD
- DM2_KernelPCA_DataImputationSVD
