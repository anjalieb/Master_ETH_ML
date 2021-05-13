import scipy as sp
import scipy.linalg as linalg
import scipy.spatial.distance as distance
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_color


'''############################'''
'''Principal Component Analysis'''
'''############################'''

'''
Compute Covariance Matrix
Input: Matrix of size #samples x #features
Output: Covariance Matrix of size #features x #features
Note: Do not use scipy or numpy cov. Implement the function yourself.
      You can of course add an assert to check your covariance function
      with those implemented in scipy/numpy.
'''
def computeCov(X=None):
    Xm = X - X.mean(axis=0)
    return 1.0/(Xm.shape[0]-1)*sp.dot(Xm.T,Xm)

'''
Compute PCA
Input: Covariance Matrix
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principal component
        eigen_vectors[:,1] the second principal component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use an implemented solver 
      to solve the eigenvalue problem!
'''
def computePCA(matrix=None):
    #compute eigen values and vectors
    [eigen_values,eigen_vectors] = linalg.eig(matrix)
    #sort eigen vectors in decreasing order based on eigen values
    indices = sp.argsort(-eigen_values)
    return [sp.real(eigen_values[indices]), eigen_vectors[:,indices]]

'''
Compute PCA using SVD
Input: Data Matrix
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principal component
        eigen_vectors[:,1] the second principal component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use SciPy svd solver!
'''
def computePCA_SVD(matrix=None):
    X = 1.0/sp.sqrt(matrix.shape[0]-1) * matrix
    [L,S,R] = linalg.svd(X)
    eigen_values = S*S
    eigen_vectors = R.T
    return [eigen_values,eigen_vectors]

'''
Compute Kernel PCA
Input: data matrix, gamma and number of components to use
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principal component, etc...
Note: Do not use an already implemented Kernel PCA algorithm.
'''
def RBFKernelPCA(X,gamma=1,n_components=2):
    #1. Compute RBF Kernel, computes matrix 300x300
    sq_dists = distance.pdist(X, 'sqeuclidean')
    mat_sq_dists = distance.squareform(sq_dists)
    K = np.exp(-gamma * mat_sq_dists)  
    
        
    #2. Center kernel matrix
    Id = np.identity(300)
    ones = np.ones((300,300)) / 300
    cent_mat = Id - ones
    Kernel_centered = cent_mat @ K @ cent_mat #300x300
    
    #3. Compute eigenvalues and eigenvactors
    [eigen_values,eigen_vectors] = linalg.eig(Kernel_centered)
    
    #4. sort eigen vectors in decreasing order based on eigen values
    indices = sp.argsort(-eigen_values)
    eigen_values = sp.real(eigen_values[indices])
    eigen_vectors = sp.real(eigen_vectors[:,indices]) #use indices on eigenvectors
    
    # select first two eigenvectors and eigenvalues
    eigen_vectors = eigen_vectors[:,0:n_components]
    eigen_values = eigen_values[:n_components]
    # normalise eigenvectors
    eigen_vectors = eigen_vectors * sp.sqrt(1/eigen_values)
    
    #5. Return transformed data
    transformed = np.transpose(np.dot(eigen_vectors.T,K))
    return transformed
'''
Transform Input Data Onto New Subspace
Input: pcs: matrix containing the first x principal components
       data: input data which should be transformed onto the new subspace
Output: transformed input data. Should now have the dimensions #samples x #components_used_for_transformation
'''


def transformData(pcs=None,data=None): 
    firsttwo = pcs[1][:,0:2] #has dimensions 40x2
    trans_X = np.transpose(np.dot(firsttwo.T,data.T))
    return trans_X

'''
Compute Variance Explaiend
Input: eigen_values
Output: return vector with varianced explained values. Hint: values should be between 0 and 1 and should sum up to 1.
'''
def computeVarianceExplained(evals=None):
    return evals/evals.sum()


'''############################'''
'''Different Plotting Functions'''
'''############################'''

'''
Plot Transformed Data
Input: transformed: data matrix (#sampels x 2)
       labels: target labels, class labels for the samples in data matrix
       filename: filename to store the plot
'''
def plotTransformedData(transformed=None,labels=None,filename=None):
    plt.figure()
    ind_l = np.unique(labels)
    legend = []
    for i,label in enumerate(ind_l):
        ind = np.where(label==labels)[0]
        plot = plt.scatter(transformed[ind,0],transformed[ind,1],color=plot_color[i],alpha=0.5)
        legend.append(plot)
    plt.legend(ind_l,scatterpoints=1,numpoints=1,prop={'size':8},ncol=6,loc="upper right",fancybox=True)
    plt.xlabel("Transformed X Values")
    plt.ylabel("Transformed Y Values")
    plt.grid(True)
    #Save File
    if filename!=None:
       plt.savefig(filename)

'''############################'''
'''Data Preprocessing Functions'''
'''############################'''

'''
Data Normalisation (Zero Mean, Unit Variance)
'''
def dataNormalisation(X=None):
    Xm = X - X.mean(axis=0)
    return Xm/np.std(Xm,axis=0)

'''
Substract Mean from Data (zero mean)
'''
def zeroMean(X=None):
    return X - X.mean(axis=0)

#    /)/)
#   ('.')
# (")(")0


