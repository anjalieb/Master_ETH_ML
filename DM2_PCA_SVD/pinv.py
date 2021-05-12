import numpy as np
import numpy.linalg as linalg

'''############################'''
'''Moore Penrose Pseudo Inverse'''
'''############################'''

'''
Compute Moore Penrose Pseudo Inverse
Input: X: matrix to invert
       tol: tolerance cut-off to exclude tiny singular values (default=1e15)
Output: Pseudo-inverse of X.
Note: Do not use scipy or numpy pinv method. Implement the function yourself.
      You can of course add an assert to compare the output of scipy.pinv to your implementation
'''
def compute_pinv(Y=None,tol=1e-15):
    # set full_matrices to false, otherwise dimensions don't fit
    U, s, Vh = linalg.svd(Y,full_matrices=False) 
    # transform s (vector) into matrix
    s = np.diag(s)
    # get the indices of entries lower than tol
    super_threshold_indices = s < tol
    # set those entries to 0
    s[super_threshold_indices] = 0
    # set the zero entries to inf in order to compute the inverse
    s[s==0]=np.inf
    s = 1./s
    # last step
    mp_inv = Vh.T @ s @ U.T
    return mp_inv

    
    