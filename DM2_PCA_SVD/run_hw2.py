#import all necessary functions
from utils import *
from pca import *
from pinv import *
#import scipy.linalg as linalg

'''
Main Function
'''
if __name__ in "__main__":
    #Initialise plotting defaults
    initPlotLib()

    ##################
    #Exercise 2:

    #Get Iris Data
    data = loadIrisData()
    
    
    #Perform a PCA using covariance matrix and eigen-value decomposition
    #1. Compute covariance matrix
    covmat = computeCov(data.data)

    #2. Compute PCA by computing eigen values and eigen vectors
    pcs = computePCA(covmat)
   
    #3. Transform input data onto a 2-dimensional subspace using the first two PCs
    transX = transformData(pcs,data.data)
    
    #4. Plot transformed data and highlight the three different sample classes
    plotTransformedData(transX, data.target,filename="transformed data.pdf")
 
    #5. How much variance can be explained with each principle component? 
    
    print("Variance Explained PCA: ")
    var = computeVarianceExplained(pcs)
    for i in range(var.shape[0]):
        print("PC %d: %.2f"%(i+1,var[i]))
    plotCumSumVariance(var)


    #Perform a PCA using SVD
    #1. Normalise data by substracting the mean
    ourdata = data.data
    X_center = zeroMean(ourdata)

    #2. Compute PCA by computing eigen values and eigen vectors
    pcs_svd = computePCA_SVD(X_center)
 
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    transX_svd = transformData(pcs_svd,data.data)

    #4. Plot your transformed data and highlight the three different sample classes
    plotTransformedData(transX_svd, data.target, filename = "transformed svd data.pdf")
 
    #5. How much variance can be explained with each principle component?

    print("Variance Explained SVD: ")
    var = computeVarianceExplained(pcs_svd)
    for i in range(var.shape[0]):
       print("PC %d: %.2f"%(i+1,var[i]))
    plotCumSumVariance(var)


    #Exercise 3
    #1. Compute the Moore-Penrose Pseudo-Inverse on the Iris data
    mp_inv = compute_pinv(data.data,tol=1e-15)
    # compare with the inbuilt Moore-Penrose Pseudo-Inverse function:
    # pi = linalg.pinv2(data.data)
    
    #2. Check Properties
    print("\nChecking status exercise 3:")
    status = False
    if np.allclose(data.data,np.dot(data.data,np.dot(mp_inv,data.data))):
        status = True
    print(f"X X^+ X = X is {status}")
    
    status = False
    if np.allclose(mp_inv,np.dot(mp_inv,np.dot(data.data,mp_inv))):
        status = True
    print(f"X^+ X X^+ = X^+ is {status}")
