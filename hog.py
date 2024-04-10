import numpy as np
from tqdm import tqdm

class HOGFeatureExtractor:
    def __init__(self, cell_size=(8, 8), block_size=(2, 2), nbins=9):
        self.cell_size = cell_size
        self.block_size = block_size
        self.nbins = nbins

    def _calculate_gradients(self, channel):
        gx = np.gradient(channel, axis=1)
        gy = np.gradient(channel, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx) * (180 / np.pi) % 180  # Convert radians to degrees
        return magnitude, direction

    def _calculate_histogram(self, magnitude, direction):
        cell_height, cell_width = self.cell_size
        nbins = self.nbins
        hist = np.zeros((magnitude.shape[0] // cell_height, magnitude.shape[1] // cell_width, nbins))

        for i in range(hist.shape[0]):
            for j in range(hist.shape[1]):
                for p in range(cell_height):
                    for q in range(cell_width):
                        weight = magnitude[i * cell_height + p, j * cell_width + q]
                        angle = direction[i * cell_height + p, j * cell_width + q]
                        bin_index = int(angle / (180 / nbins))
                        hist[i, j, bin_index] += weight

        return hist

    def _block_normalization(self, block):
        epsilon = 1e-5
        normalized_block = block / np.sqrt(np.sum(block**2) + epsilon**2)
        return normalized_block.flatten()

    def extract_features(self, rgb_image,unflatten):
        features = []
        for i in range(rgb_image.shape[2]):  # Loop through each channel
            magnitude, direction = self._calculate_gradients(rgb_image[:, :, i])
            hist = self._calculate_histogram(magnitude, direction)

            block_height, block_width = self.block_size
            for row in range(hist.shape[0] - block_height + 1):
                for col in range(hist.shape[1] - block_width + 1):
                    block = hist[row:row+block_height, col:col+block_width, :]
                    if unflatten :
                        features.append(self._block_normalization(block))
                    else :
                        features.extend(self._block_normalization(block))

        return np.array(features)
    
    def transform(self,X,unflatten=False):
        """tranforms images into un/flattened feature maps
        X : set of images of shape (n_images,nx,ny,channels)
        """
        n = X.shape[0]
        features=[]
        for i in tqdm(range(n)):
            features.append(self.extract_features(X[i],unflatten))
        return np.array(features)