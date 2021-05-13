import scipy as sp
import scipy.linalg as linalg
import matplotlib.pylab as pl
import numpy as np

from utils import plot_color

'''############################'''
'''Principle Component Analyses'''
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
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principle component
        eigen_vectors[:,1] the second principle component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use an implemented solver 
      to solve the eigenvalue problem!
'''
def computePCA(matrix=None):
    # compute eigen values and vectors
    [eigen_values,eigen_vectors] = linalg.eig(matrix)
    # sort eigen vectors in decreasing order based on eigen values
    indices = sp.argsort(-eigen_values)
    # sort the eigen_values based on thethe decreasing indices 
    eigen_values = eigen_values[indices]
    # sort eigenvectors correspondingly
    eigen_vectors = eigen_vectors[:,indices] 
    
    pcs = [sp.real(eigen_values),eigen_vectors]
    return pcs

'''
Compute PCA using SVD
Input: Data Matrix
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principle component
        eigen_vectors[:,1] the second principle component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use SciPy svd solver!
'''
def computePCA_SVD(X_centered):
    #compute svd of transposed mean centered data, transposed is necessary to get right dimensions for multiplication
    U, s, Vh = linalg.svd(X_centered.T)
    # the squared sigma and divison by n corresponds to the eigenvalues
    eigen_values = s*s.T
    # divide by n, here: n = 4
    eigen_values = eigen_values/4 
    indices = sp.argsort(-eigen_values)
    eigen_values = eigen_values[indices]
    # the left singular vectors correspond to the eigenvectors
    eigen_vectors = U
    # sort them
    eigen_vectors = eigen_vectors[:,indices] 
    
    pcs = [eigen_values,eigen_vectors]
    return pcs


'''
Transform Input Data Onto New Subspace
Input: pcs: matrix containing the first x principle components
       data: input data which should be transformed onto the new subspace
Output: transformed input data. Should now have the dimensions #samples x #components_used_for_transformation
'''
def transformData(pcs,dataa):
    firsttwo = pcs[1][:,0:2]
    return np.transpose(np.dot(firsttwo.T,dataa.T))

'''
Compute Variance Explaiend
Input: eigen_values
Output: return vector with varianced explained values. Hint: values should be between 0 and 1 and should sum up to 1.
'''
def computeVarianceExplained(pcs):
    evals = pcs[0][:]
    # evals is a list, comvert it to np array  
    evals = np.array(evals)  
    var = evals/evals.sum()
    var = np.array(var)
    return var


'''############################'''
'''Different Plotting Functions'''
'''############################'''

'''
Plot Cumulative Explained Variance
Input: var: variance explained vector
       filename: filename to store the file
'''
def plotCumSumVariance(var=None,filename=None):
    pl.figure()
    pl.plot(sp.arange(var.shape[0]),sp.cumsum(var)*100)
    pl.xlabel("Principle Component")
    pl.ylabel("Cumulative Variance Explained in %")
    pl.grid(True)
    #Save File
    if filename!=None:
        pl.savefig(filename)

'''
Plot Transformed Data
Input: transformed: data matrix (#sampels x 2)
       labels: target labels, class labels for the samples in data matrix
       filename: filename to store the plot
'''
def plotTransformedData(transformed=None,labels=None,filename=None):
    pl.figure()
    ind_l = sp.unique(labels)
    legend = []
    for i,label in enumerate(ind_l):
        ind = sp.where(label==labels)[0]
        plot = pl.scatter(transformed[ind,0],transformed[ind,1],color=plot_color[i],alpha=0.5)
        legend.append(plot)
    pl.legend(ind_l,scatterpoints=1,numpoints=1,prop={'size':8},ncol=6,loc="upper right",fancybox=True)
    pl.xlabel("Transformed X Values")
    pl.ylabel("Transformed Y Values")
    pl.grid(True)
    #Save File
    if filename!=None:
       pl.savefig(filename)

'''############################'''
'''Data Preprocessing Functions'''
'''############################'''

'''
Data Normalisation (Zero Mean, Unit Variance)
'''
def dataNormalisation(X=None):
    Xm = X - X.mean(axis=0)
    return Xm/sp.std(Xm,axis=0)

'''
Substract Mean from Data (zero mean)
'''
def zeroMean(X=None):
    # compute mean of each feature (col) and substract it from data
    Xm = X - X.mean(axis=0) 
    return Xm
