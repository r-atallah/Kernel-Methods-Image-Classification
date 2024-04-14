import numpy as np
from tqdm import tqdm
import cv2

class SIFTFeatureExtractor:
    def __init__(self, num_features=0, num_octave_layers=3, contrast_threshold=0.001, edge_threshold=10, sigma=1.6):
        self.num_features = num_features
        self.num_octave_layers = num_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma

    def extract_features(self, image):
        # Convert the image to 8-bit unsigned integer (CV_8U) depth
        image = cv2.convertScaleAbs(image)

        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create(
            nfeatures=self.num_features,
            nOctaveLayers=self.num_octave_layers,
            contrastThreshold=self.contrast_threshold,
            edgeThreshold=self.edge_threshold,
            sigma=self.sigma
        )
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        return keypoints, descriptors
    def transform(self,X,unflatten=False):
        """
        tranforms images into un/flattened feature maps
        X : set of images of shape (n_images,nx,ny,channels)
        If unflatten, Returns a list of arrays where each one is the keypoints descriptors found
        if flatten, returns an array of shape (n_images, -1 , dim_descriptor) with padding to make sure it fits in one array
        """
        n = X.shape[0]
        features = []
        max_keypoints = 0

        for i in tqdm(range(n)):
            keypoints, descriptors = self.extract_features(X[i])
            if descriptors is not None:
                features.append(descriptors)
                max_keypoints = max(max_keypoints, descriptors.shape[0])
            else:
                print(f"No keypoints found for image {i}")
            
        if unflatten:
            return features
        else:
            all_descriptors = np.zeros((n, max_keypoints, features[0].shape[1]))
            for i, desc in enumerate(features):
                all_descriptors[i, :desc.shape[0], :] = desc

            return all_descriptors