from sklearn.preprocessing import LabelBinarizer
from scipy.linalg import solve
import numpy as np
from tqdm import tqdm

from kernels import GaussianKernel,LinearKernel,LaplacianRBFKernel,HellingerKernel,SublinearRBFKernel,GaussianKernel_orientation

class KernelRidgeClassifier():
    def __init__(self, C=1.0, kernel='rbf', gamma=10):
        self.C = C
        if kernel == 'linear':
            self.kernel = LinearKernel()
        elif kernel == 'rbf':
            self.kernel = GaussianKernel(1/np.sqrt(2*gamma))
        elif kernel =='laplacian_rbf':
            self.kernel = LaplacianRBFKernel(1/np.sqrt(2*gamma))
        self.gamma = gamma
        self.K = None
        self.alpha = None
        self.support = None
    def fit(self, X, y):
        self.support = X
        # map labels in {-1, 1}
        Y = LabelBinarizer(pos_label=1, neg_label=-1).fit_transform(y)
        # initialize kernel
        self.K = self.kernel.build_K(X,X)
        # compute first term
        diag = np.zeros_like(self.K)
        np.fill_diagonal(diag, self.C * len(X))
        self.K += diag
        # compute coefficients for each class, one-vs-all
        self.alpha = []
        for c in tqdm(sorted(set(y))):
            self.alpha.append(solve(self.K, Y[:, c], assume_a='pos'))
        self.alpha = np.array(self.alpha).T

    def predict(self, X):
        return np.argmax(self.kernel.build_K(X,self.support) @ self.alpha,axis=1).reshape(-1,1)
        """for x in tqdm(X):
            similarity = self.K.similarity(x)
            preds.append(np.argmax([np.dot(alpha, similarity) for alpha in self.alpha]))"""