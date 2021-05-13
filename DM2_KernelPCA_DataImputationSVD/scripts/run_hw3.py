"""
Course  : Data Mining II (636-0019-00L)
"""
from utils import *
from pca import *
from impute import *

import scipy.misc as misc
import scipy.ndimage as sn
import numpy as np 
from sklearn.datasets import make_moons
import matplotlib
import matplotlib.pyplot as plt
import seaborn_image as isns
'''
Main Function
'''
if __name__ in "__main__":
    
    # font size in plots:
    fs = 10
    matplotlib.rcParams['font.size'] = fs
    
    #################
    #Exercise 1:
    ranks = np.arange(1,30) #30
    
    # get image data:
    img = misc.ascent() 
    X = sn.rotate(img, 180) #we rotate it for displaying with seaborn
    
    #generate data matrix with 80% missing values
    X_missing = randomMissingValues(X,per=0.60)
    
    #plot data for comparison
    fig, ax = plt.subplots(1, 2)
    isns.imgplot(X, ax=ax[0], cbar=False)

    isns.imgplot(X_missing, ax=ax[1], cbar=False)
    ax[0].set_title('Original')
    ax[1].set_title(f'60% missing data')
    plt.savefig('exercise1_1.pdf')
     
    #Impute data with optimal rank r
    [X_imputed,r,testing_errors] = svd_imputation_optimised(
        X=X_missing,
        ranks=ranks,
        test_size=0.3
    )

    #plot data for comparison
    fig, ax = plt.subplots(1, 3)
    isns.imgplot(X, ax=ax[0], cbar=False)
    isns.imgplot(X_missing, ax=ax[1], cbar=False)
    isns.imgplot(X_imputed, ax=ax[2], cbar=False)
    ax[0].set_title('Original', fontsize=fs)
    ax[1].set_title(f'60% missing data', fontsize=fs)
    ax[2].set_title('Imputed', fontsize=fs)
    plt.savefig("exercise1_2.pdf")

    
    
    #Plot testing_error and highlight optimial rank r
    plt.figure()
    plt.plot(ranks[18], testing_errors[18], 'g*')
    plt.plot(ranks[0:18],testing_errors[0:18], 'ro')
    plt.plot(ranks[19:],testing_errors[19:], 'ro')
    plt.xlabel('ranks')
    plt.ylabel('MSE between test data and imputed data')
    plt.title('tested ranks with corresponding MSE')
    plt.savefig('exercise1_3.pdf')

    #Exercise 2
    #load data
    [X,y] = make_moons(n_samples=300,noise=None)
    
    #Perform a PCA
    #1. Compute covariance matrix
    X_covmat = computeCov(X)
    #2. Compute PCA by computing eigen values and eigen vectors
    pca = computePCA(X_covmat)
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    trans_X = transformData(pca,X)
    #4. Plot your transformed data and highlight the sample classes.
    plotTransformedData(trans_X,y,filename="Ex.2 transformed data.pdf")
    #5. How much variance can be explained with each principle component?
    eigen_vals = pca[0][:]
   
    sp.set_printoptions(precision=2)
    var = computeVarianceExplained(eigen_vals)
    print("Variance Explained Exercise 2a: ")
    for i in range(var.shape[0]):
        print("PC %d: %.2f"%(i+1,var[i]))

    #TODO:
    #1. Perform Kernel PCA
    #2. Plot your transformed data and highlight the sample classes
    #3. Repeat the previous 2 steps for gammas [1,5,10,20] and compare the results.
    gam_vec = [1,5,10,20]
    for g in gam_vec:

        trans_kernel = RBFKernelPCA(X,gamma=g,n_components=2)
        plotTransformedData(trans_kernel,y,filename='Ex.2 transformed kernel data.pdf')
            
