import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

def logsumexp(v):
    log_sum = np.log(np.sum(np.exp(v)))
    return v - log_sum #(log(a/b) = log(a) - log(b))

def posterior(x, pi, mu, sigma):
    K = len(pi)
    likelihoods = np.empty(K)
    d = len(x)
    for i in range(K):
        x_temp = x - mu[i]
        sigma_inv = np.linalg.inv(sigma[i])
        likelihoods[i] = np.log(pi[i]) - 0.5 * d * np.log(2 * np.pi) \
                        - 0.5 * np.sum(np.log(np.linalg.eigvals(sigma[i]))) \
                        - 0.5 * (x_temp.T @ sigma_inv @ x_temp)
    logsum = logsumexp(likelihoods)
    #posterior_probs = np.exp(likelihoods - logsum)
    return np.exp(logsum)

class FisherVector:
    """
    To calculate the signatures and store the fisher vector information
    """
    def __init__(self, nclasses, dim, w, mu, sigma):
        """
        nclasses: number of classes from gmm
        dim: dimension of local descriptors
        w: weights of gmm
        mu: means of different classes
        sigma: diagonal values of the covariance matrix, represented by a 2d numpy array
        (Note that our sigma is sigma^2 in the paper)
        s0: statistics 0, 1d array of length nclassses
        s1: statistics 1, 2d array of shape (nclasses, dim)
        s2: statistics 2, 2d array of shape (nclasses, dim)
        fv: Fisher vector, 1d array of length nclasses * (2 * dim + 1)
        """
        self.nclasses = nclasses
        self.dim = dim
        self.w = w
        self.mu = mu
        self.sigma = sigma
        self.s0 = None
        self.s1 = None
        self.s2 = None
        self.fv = None
        
    def _compute_statistics(self, X):
        n_descriptors = len(X)
        gamma = np.zeros((self.nclasses, n_descriptors))
        cov = np.zeros((self.nclasses, self.dim, self.dim))
        for k in range(self.nclasses):
            cov[k] = np.diag(self.sigma[k])
        for t in range(n_descriptors):
            gamma[:,t] = posterior(X[t], self.w, self.mu, cov)
            
        self.s0 = np.sum(gamma, axis=1)
        self.s1 = np.sum(X[np.newaxis,: , :] * gamma[:, :, np.newaxis], axis=1)
        self.s2 = np.sum(X[np.newaxis,: , :]**2 * gamma[:, :, np.newaxis], axis=1)
        
    def _compute_signature(self, X):
        n = len(X)
        self.fv = np.zeros(self.nclasses * (2 * self.dim + 1))
        signature_temp_0 = (self.s0 - n * self.w) / np.sqrt(self.w)
        self.fv[:self.nclasses] = signature_temp_0
        signature_temp_1 = (self.s1 - self.mu * self.s1) / (np.sqrt(self.w[:, np.newaxis] * self.sigma))
        self.fv[self.nclasses:self.nclasses + self.nclasses * self.dim] = signature_temp_1.ravel()                            
        signature_temp_2 = (self.s2 ** 2 - 2 * self.mu * self.s1 + 
                            (self.mu ** 2 - self.sigma) * self.s0[:,np.newaxis]) / (np.sqrt(2 * self.w)[:,np.newaxis] * self.sigma)
        self.fv[self.nclasses + self.nclasses * self.dim:] = signature_temp_2.ravel()
        
    def _normalize(self):
        self.fv = np.sign(self.fv) * np.sqrt(np.abs(self.fv))
        self.fv /= np.linalg.norm(self.fv, ord=2)
        
    def predict(self, X):
        """
        X of shape (n_descriptors,dim) is the set of local descriptors for one image
        """
        self._compute_statistics(X)
        self._compute_signature(X)
        self._normalize()
        return self.fv

class FisherVectorExtractor:
    def __init__(self, nclasses=128, gmm_niter=10):
        """
        local_feature: can be either 'hog' or 'sift'
        nclasses: number of classes for the GMM
        gmm_niter : max iters for gmm
        """
        self.nclasses = nclasses
        self.gmm_niter = gmm_niter
        self.gmm=None
    def train(self, X):
        """
        trains the fisher vector on a set of local descriptors
        X : list of length (n_images) of unflattened local descriptors (n_descriptors,dim_descriptors), because we'll perform gmm over it 
        """
        n = len(X)
        descriptors = np.concatenate(X, axis=0)
        features = []
        
        gmm = GaussianMixture(n_components=self.nclasses,init_params='k-means++',max_iter=self.gmm_niter,covariance_type='diag')
        gmm.fit(descriptors)
        fisher_vector = FisherVector(self.nclasses, descriptors.shape[-1], gmm.weights_, gmm.means_, gmm.covariances_)

        for i in tqdm(range(n)):
            features.append(fisher_vector.predict(X[i]))
        self.gmm = gmm
        return np.array(features)
    
    def predict(self, X):
        """
        predicts the fisher vector using the set of local descriptors for every image and the trained gmm
        X : set of unflattened local descriptors (n_images,n_descriptors,dim_descriptors)
        """
        n = X.shape[0]
        features = []
        
        fisher_vector = FisherVector(self.nclasses, X.shape[-1], self.gmm.weights_, self.gmm.means_, self.gmm.covariances_)

        for i in tqdm(range(n)):
            features.append(fisher_vector.predict(X[i]))

        return np.array(features)