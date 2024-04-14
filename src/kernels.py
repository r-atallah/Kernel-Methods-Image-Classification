from tqdm import tqdm
import numpy as np

class Kernel:
    def __init__(self):
        self.name = None

    def calc(self, x, y):
        raise NotImplementedError("Should be implemented")

    def build_K(self, X, Y=None):
        if Y is None:
            Y = X
        n = X.shape[0]
        m = Y.shape[0]
        K = np.zeros((n, m))

        for i in tqdm(range(n)):
            for j in range(m):
                K[i, j] = self.calc(X[i, :], Y[j, :])
        return K

class LinearKernel(Kernel):
    def __init__(self):
        self.name = 'linear'

    def calc(self, x, y):
        return np.dot(x, y)

    def build_K(self, X, Y=None):
        if Y is None:
            Y = X
        return np.dot(X, Y.T)

class GaussianKernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma
        self.name = 'gaussian_%.5f' % sigma

    def calc(self, x, y):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * self.sigma ** 2))

    def build_K(self, X, Y=None,verbose=True):
        if Y is None:
            Y = X
        n = X.shape[0]
        m = Y.shape[0]
        K = np.zeros((n, m))
        if verbose :
            for i in tqdm(range(m)):
                K[:, i] = np.linalg.norm(X - Y[i, :], axis=1) ** 2
        else :
            for i in range(m):
                K[:, i] = np.linalg.norm(X - Y[i, :], axis=1) ** 2
        K /= 2 * self.sigma ** 2
        return np.exp(-K)

class GaussianKernel_orientation(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma
        self.name = 'gaussian_angle_%.5f' % sigma

    def calc(self, x, y):
        aux = (np.sin(x) - np.sin(y)) ** 2 + (np.cos(x) - np.cos(y)) ** 2
        return np.exp(-aux / (2 * self.sigma ** 2))

    def build_K(self, X, Y=None):
        if Y is None:
            Y = X
        n = X.shape[0]
        m = Y.shape[0]
        X2 = np.concatenate((np.sin(X), np.cos(X)), axis=1)
        Y2 = np.concatenate((np.sin(Y), np.cos(Y)), axis=1)
        K = np.zeros((n, m))

        for i in tqdm(range(m)):
            K[:, i] = np.linalg.norm(X2 - Y2[i, :], axis=1) ** 2
        K /= 2 * self.sigma ** 2
        return np.exp(-K)

class HistogramIntersectionKernel(Kernel):
    def __init__(self, beta):
        self.beta = beta
        self.name = 'histogram_intersection'

    def calc(self, x, y):
        return np.sum(np.minimum(x ** self.beta, y ** self.beta))

class LaplacianRBFKernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma
        self.name = 'laplacian_%.5f' % sigma

    def calc(self, x, y):
        return np.exp(-np.sum(np.abs(x - y)) / self.sigma**2)

    def build_K(self, X, Y=None):
        if Y is None:
            Y = X
        n = X.shape[0]
        m = Y.shape[0]
        K = np.zeros((n, m))

        for i in tqdm(range(m)):
            K[:, i] = np.sum(np.abs(X - Y[i, :]), axis=1)
        K /= self.sigma ** 2
        return np.exp(-K)

class SublinearRBFKernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma
        self.name = 'sublinear_%.5f' % sigma

    def calc(self, x, y):
        return np.exp(-np.sum(np.abs(x - y))**0.5 / self.sigma**2)

class HellingerKernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma
        self.name = 'hellinger_%.5f' % sigma

    def calc(self, x, y):
        return np.sum(np.sqrt(x * y))
