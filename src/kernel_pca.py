import numpy as np

from kernels import GaussianKernel,LinearKernel,LaplacianRBFKernel,HellingerKernel,SublinearRBFKernel,GaussianKernel_orientation

class KernelPCA:
    
    def __init__(self,kernel, r=2):                             
        self.kernel = kernel          # <---
        self.alpha = None # Matrix of shape N times d representing the d eingenvectors alpha corresp
        self.lmbda = None # Vector of size d representing the top d eingenvalues
        self.support = None # Data points where the features are evaluated
        self.r =r ## Number of principal components
    def fit(self, X):
        # assigns the vectors
        self.support = X
        ###
        N= self.support.shape[0]
        K = self.kernel.build_K(X,X)
        row_sum = np.sum(K, axis=1)
        col_sum = np.sum(K, axis=0)
        total_sum = np.sum(K)
        G = (K - (1/N) * row_sum[:, np.newaxis] - (1/N) * col_sum + (1/N**2) * total_sum)
        eigenvalues, eigenvectors = np.linalg.eig(G)
        idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[idx]
        sorted_eigenvectors = eigenvectors[:, idx]
        top_eigenvectors = sorted_eigenvectors[:, :self.r]
        norms = np.sqrt(np.diag(top_eigenvectors.T @ G @ top_eigenvectors))
        top_eigenvectors = top_eigenvectors / norms[np.newaxis , :]
        ###
        #self.lmbda = 
        self.alpha = top_eigenvectors
        
        #constraints = ({})
        # Maximize by minimizing the opposite
        return K @ self.alpha - (1/N) * K @ np.ones(N) @ np.sum(self.alpha,axis=1)
        
    def predict(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        N= self.support.shape[0]
        K = self.kernel.build_K(x,self.support)   
        return K @ self.alpha - (1/N) * np.outer(K @ np.ones(N) , np.sum(self.alpha,axis=0))
    
