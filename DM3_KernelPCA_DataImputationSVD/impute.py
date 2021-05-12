"""
Course  : Data Mining II (636-0019-00L)
"""
import scipy as sp
import numpy as np
import scipy.linalg as linalg
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


'''
Impute missing values using the mean of each feature
Input: X: data matrix with missing values (sp.nan) of size n x m, 
          where n is the number of samples and m the number of features
Output: D_imputed (n x m): Mean imputed data matrix
'''
def mean_imputation(X=None):
    D_imputed = X.copy()
    #Impute each missing entry per feature with the mean of each feature
    for i in range(X.shape[1]):
        feature = X[:,i]
        #get indices for all non-nan values
        indices = sp.where(~sp.isnan(feature))[0]
        #compute mean for given feature
        mean = sp.mean(feature[indices])
        #get nan indices
        nan_indices = sp.where(sp.isnan(feature))[0]
        #Update all nan values with the mean of each feature
        D_imputed[nan_indices,i] = mean
    return D_imputed

'''
Impute missing values using SVD
Input: X: data matrix with missing values (sp.nan) of size n x m,
          where n is the number of sampkes and m the number of features
       rank: rank for the rank-r approximation of the orginal data matrix
       tol: precision tolerance for iterative optimisier to stop (default=0.1). The smaller the better!
       max_iter: maximum number of iterations for optimiser (default=100)
Output: D_imputed (n x m): Mean imputed data matrix
'''
def svd_imputation(X=None,rank=None,tol=.1,max_iter=100):
    #get all nan indices
    nan_indices = sp.where(sp.isnan(X))
    #initialise all nan entries with the a mean imputation
    D_imputed = mean_imputation(X)
    #repeat approximation step until convergance
    for i in range(max_iter):
        D_old = D_imputed.copy()
        #SVD on mean_imputed data
        [L,d,R] = linalg.svd(D_imputed)
        #compute rank r approximation of D_imputed
        D_r = sp.matrix(L[:,:rank])*sp.diag(d[:rank])*sp.matrix(R[:rank,:])
        #update imputed entries according to the rank-r approximation
        imputed = D_r[nan_indices[0],nan_indices[1]]
        #print(imputed)
        D_imputed[nan_indices[0],nan_indices[1]] = sp.asarray(imputed)[0]
        #use Frobenius Norm to compute similarity between new matrix and the latter approximation
        fnorm = linalg.norm(D_old-D_imputed,ord="fro")
        if fnorm<tol:
            print("\t\t\t[SVD Imputation]: Converged after %d iterations"%(i+1))
            break
        if (i+1)>=max_iter:
            print("\t\t\t[SVD Imputation]: Maximum number of iterations reached (%d)"%(i+1))
    return D_imputed

'''
Find Optimal Rank-r Imputation
Input: X: data matrix with missing values (sp.nan) of size n x m,
          where n is the number of samples and m the number of features
       ranks: int array with r values to use for optimisation
       test_size: float between 0.0 and 1.0 and represent the proportion of the
                  non-nan values of the data matrix to use for optimising the rank r
                  (default: 0.25)
       random_state: Pseudo-random number generator state used for random splitting (default=0)
       return_optimal_rank: return optimal r (default: True)
       return_errors: return array of testing-errors (default: True)
Output: X_imputed: imputed data matrix using the optimised rank r
        r: optimal rank r [if flag is set]
        errors: array of testing-errors [if flag is set]
'''
def svd_imputation_optimised(X=None,ranks=None,
                             test_size=0.3,random_state=0,
                             return_optimal_rank=True,return_errors=True):
    #init variables
    sp.random.seed(random_state)
    testing_errors = []
    optimal_rank = sp.nan
    minimal_error = sp.inf



    #1. Get all non-nan indices
    indices = sp.where(~sp.isnan(X))
    co = len(indices[0])
    
    #2. Use "test_size" % of the non-missing entries as test data
    test = sp.random.choice(range(co), int(test_size*co),replace=False)
    test_ind = (indices[0][test],indices[1][test])
    
    
    #3. Create a new training data matrix
    new_train = X.copy()
    new_train[test_ind[0],test_ind[1]] = sp.nan
    

    previous = sp.inf
    
    #4. Find optimal rank r by minimising the Frobenius-Norm using the train and test data 
    for rank in ranks:
        print("\tTesting rank %d..."%(rank))
        #4.1 Impute Training Data
        train_imputed = svd_imputation(new_train,rank,tol=.1,max_iter=100) 
        X_pred = train_imputed[test_ind[0],test_ind[1]]
        X_true = X[test_ind[0],test_ind[1]]
        
        #4.2 Compute the mean squared error of imputed test data with original test data and store error in array
        error = mean_squared_error(X_true,X_pred) 
        testing_errors.append(error)
        
        print("\t\tMean Squared Error: %.2f"%error)
        #4.3 Update rank if necessary)
        
        if error < previous:
            optimal_rank = rank
            minimal_error = error
            previous = error
            
    #5. Use optimal rank for imputing the "original" data matrix
    print("Optimal Rank: %f (Mean-Squared Error: %.2f)"%(optimal_rank,minimal_error))
    X_imputed = svd_imputation(X,optimal_rank,tol=.1,max_iter=100)
    
    # for answering Ex. 1c, the original data matrix was imputed using a too high and a too low rank
    #optimal_rank = 8
    #What would happen if a too large rank r is chosen to impute the data?
    #X_imputed = svd_imputation(X,optimal_rank,tol=.1,max_iter=100)
    #return data
    if return_optimal_rank==True and return_errors==True:
        return [X_imputed,optimal_rank,testing_errors]
    elif return_optimal_rank==True and return_errors==False:
        return [X_imputed,optimal_rank]
    elif return_optimal_rank==False and return_errors==True:
        return [X_imputed,testing_errors]
    else:
        return X_imputed
