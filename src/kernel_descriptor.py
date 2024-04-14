import numpy as np
from tqdm import tqdm
from kernels import GaussianKernel,LinearKernel,LaplacianRBFKernel,HellingerKernel,SublinearRBFKernel,GaussianKernel_orientation
from kernel_pca import KernelPCA

class FeatureMap():
    def __init__(self, kernel):
        self.kernel = kernel
        self.support = None
        self.G = None
        self.dim = None

    def fit(self, X):
        self.support = X
        self.dim = X.shape[0]
        K = self.kernel.build_K(X)
        Kinv = np.linalg.inv(K)
        self.G = np.linalg.cholesky(Kinv)
        #self.G = G.real

    def predict(self, X):
        K = self.kernel.build_K(X, self.support)
        return K @ self.G
    
class KernelDecriptorExtractor: #Color and Grad kernel maps
    def __init__(self, kpca_sigma=0.4 ,kpca_components=200, gamma_o=5, gamma_c=4, gamma_b=2, gamma_p=3,
                    grid_o_dim=25, grid_c_shape=(5, 5, 5), grid_p_shape=(5,5),
                    epsilon_g=0.8, epsilon_s=0.2):
        
        self.epsilon_g = epsilon_g
        self.epsilon_s = epsilon_s
        ## Define Kernels
        k_o = GaussianKernel_orientation(1 / np.sqrt(2 * gamma_o))
        k_c = GaussianKernel(1 / np.sqrt(2 * gamma_c)) #convention used by the article
        k_b = GaussianKernel(1 / np.sqrt(2 * gamma_b))
        k_p = GaussianKernel(1 / np.sqrt(2 * gamma_p))
        ## Define feature maps (find G using sampled sampled vectors) and projections (phi_.^tilde(xi)):
        # position
        self.map_p = FeatureMap(k_p)
        x_values = np.linspace(0, 1, grid_p_shape[0])
        y_values = np.linspace(0, 1, grid_p_shape[1])
        X = np.array(np.meshgrid(x_values, y_values)).T.reshape(-1,2)
        self.map_p.fit(X)
        X_p = self.map_p.predict(self.map_p.support)
        # color
        self.map_c = FeatureMap(k_c)
        r_values = np.linspace(0, 1, grid_c_shape[0])
        g_values = np.linspace(0, 1, grid_c_shape[1])
        b_values = np.linspace(0, 1, grid_c_shape[2])
        X = np.array(np.meshgrid(r_values, g_values, b_values)).T.reshape(-1, 3)
        self.map_c.fit(X)
        X_c = self.map_c.predict(self.map_c.support)
        # orientation
        self.map_o = FeatureMap(k_o)
        X = np.linspace(-np.pi, np.pi, grid_o_dim + 1)[:-1]
        X = X[:, np.newaxis]
        self.map_o.fit(X)
        X_o = self.map_o.predict(self.map_o.support)
        ## Compute tensor products between basis eigenvectors (o&p + c&p)
        X_op = np.kron(X_o, X_p)
        X_cp = np.kron(X_c, X_p)
        ## fit Kernel PCA 
        kpca_kernel = GaussianKernel(kpca_sigma)
        self.kpca_op = KernelPCA(kpca_kernel,kpca_components)
        self.kpca_cp = KernelPCA(kpca_kernel,kpca_components)
        self.kpca_op.fit(X_op)
        self.kpca_cp.fit(X_cp)
        
    def K_grad_map_image(self, image, patch_size, stride,unflatten):
        """compute K_grad over patches and perform kpca"""
        nx, ny, nchannels = image.shape
        # Compute magnitude and angle of gradients
        dI_dx = np.roll(image, -1, axis=0) - np.roll(image, 1, axis=0) 
        dI_dy = np.roll(image, -1, axis=1) - np.roll(image, 1, axis=1) 
        Ig_magnitude = np.sqrt(dI_dx**2 + dI_dy**2)
        Ig_angle = np.arctan2(dI_dy, dI_dx) 
        # compute position feature (same for all patches)
        x_values = np.linspace(0, 1, patch_size[0])
        y_values = np.linspace(0, 1, patch_size[1])
        X_p = np.array(np.meshgrid(x_values, y_values)).T.reshape(-1,2)
        X_p = self.map_p.predict(X_p)
        # compute orientation feature by patch and then grad feature
        op_dim = self.map_o.dim * self.map_p.dim
        features = []
        for sx in range(0, nx - patch_size[0] + 1, stride[0]):
            for sy in range(0, ny - patch_size[1] + 1, stride[1]):
                # compute patch norm
                norm = np.sqrt(self.epsilon_g + np.sum(Ig_magnitude[sx:sx + patch_size[0], sy:sy + patch_size[1]] ** 2))
                # Extract patch angle and project on angle basis eigenvectors
                X_o = Ig_angle[sx:sx + patch_size[0], sy:sy + patch_size[1]].reshape(-1, 1) 
                X_o = self.map_o.predict(X_o)
                # Now Compute the feature for the current patch by summing (magnitude * (o x p))
                aux = np.zeros(op_dim)
                for magnitude, x_o, x_p in zip(Ig_magnitude[sx:sx + patch_size[0], sy:sy + patch_size[1]].ravel(), X_o, X_p):
                    aux += magnitude * np.kron(x_o, x_p)
                features.append(aux / norm)
        features = np.array(features)
        # reduce dimension by kernel PCA
        if unflatten :
            return self.kpca_op.predict(features)
        return self.kpca_op.predict(features).flatten()
    
    def K_color_map_image(self, image, patch_size, stride,unflatten):
        nx, ny, _ = image.shape
        # Compute position feature (same for all patches)
        x_values = np.linspace(0, 1, patch_size[0])
        y_values = np.linspace(0, 1, patch_size[1])
        X_p = np.array(np.meshgrid(x_values, y_values)).T.reshape(-1, 2)
        X_p = self.map_p.predict(X_p)
        # compute color feature by patch and then final color feature
        cp_dim = self.map_c.dim * self.map_p.dim
        features = []
        for sx in range(0, nx - patch_size[0] + 1, stride[0]):
            for sy in range(0, ny - patch_size[1] + 1, stride[1]):
                # Compute color feature for the current patch
                X_c = image[sx:sx + patch_size[0], sy:sy + patch_size[1], :].reshape(-1, 3)
                X_c_proj = self.map_c.predict(X_c)
                # Now Compute the feature for the current patch by summing (c x p)
                aux = np.zeros(cp_dim)
                for x_c, x_p in zip(X_c_proj, X_p):
                    aux += np.kron(x_c, x_p)
                features.append(aux)
        features = np.array(features)
        # reduce dimension by kernel PCA
        if unflatten :
            return self.kpca_cp.predict(features)
        return self.kpca_cp.predict(features).flatten()
    def transform(self, X, patch_size=(16, 16), stride=(8, 8), match_kernel='gradient',unflatten=False):
        """tranforms images into un/flattened feature maps
        X : set of images of shape (n_images,nx,ny,channels)
        """
        n = X.shape[0]
        if match_kernel == 'gradient':
            X_grad = []
            for i in tqdm(range(n)):
                X_grad.append(self.K_grad_map_image(X[i,:,:,:], patch_size, stride,unflatten))
            X_grad = np.array(X_grad)
            return X_grad
        elif match_kernel == 'color':
            X_color = []
            for i in tqdm(range(n)):
                X_color.append(self.K_color_map_image(X[i,:,:,:], patch_size, stride,unflatten))
            X_color = np.array(X_color)
            return X_color