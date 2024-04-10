import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import rotate


def read_data(path):
    """
    read the challenge data X_train,X_test,Y_train
    """
    Xtr = np.array(pd.read_csv(f'{path}/Xtr.csv.zip',header=None,sep=',',usecols=range(3072))) 
    Xte = np.array(pd.read_csv(f'{path}/Xte.csv.zip',header=None,sep=',',usecols=range(3072))) 
    Ytr = np.array(pd.read_csv(f'{path}/Ytr.csv',sep=',',usecols=[1])).squeeze()
    return Xtr,Xte,Ytr

def augment_data(X, y, flip_ratio=0.2, rot_replicas=1, rot_ratio=0.2, rot_angle=None):
    """
    Augment the dataset by flipping and rotating
    """
    augmented_X = []
    augmented_y = []
    # flip
    print('Flipping data ...')
    for c in tqdm(set(y)):
        indexes = (y == c).nonzero()[0]
        chosen_indexes = np.random.choice(indexes, int(flip_ratio * len(indexes)), replace=False)
        X_to_augment = X[chosen_indexes]

        for x in X_to_augment:
            flip_x = np.array([])

            r = x[:1024].reshape([32, 32])
            g = x[1024:2048].reshape([32, 32])
            b = x[2048:].reshape([32, 32])
            tensor = np.dstack((r, g, b))

            tensor = np.fliplr(tensor)

            for channel in range(3):
                flip_x = np.append(flip_x, tensor[:, :, channel])
            augmented_X.append(flip_x)
            augmented_y.append(c)
    # rotate
    if rot_angle:
        print('Rotating data ...')
        for _ in range(rot_replicas):
            for c in tqdm(set(y)):
                indexes = (y == c).nonzero()[0]
                chosen_indexes = np.random.choice(indexes, int(rot_ratio * len(indexes)), replace=False)
                X_to_augment = X[chosen_indexes]

                for x in X_to_augment:
                    rot_x = np.array([])

                    r = x[:1024].reshape([32, 32])
                    g = x[1024:2048].reshape([32, 32])
                    b = x[2048:].reshape([32, 32])
                    tensor = np.dstack((r, g, b))

                    if np.random.random_sample() >= 0.5:
                        tensor = np.fliplr(tensor)

                    angle = np.random.randint(-rot_angle, rot_angle)
                    tensor = rotate(tensor, angle, reshape=False, mode='nearest')

                    for channel in range(3):
                        rot_x = np.append(rot_x, tensor[:, :, channel])
                    augmented_X.append(rot_x)
                    augmented_y.append(c)
    indexes = np.random.permutation(len(augmented_X))
    augmented_X = np.array(augmented_X)[indexes]
    augmented_y = np.array(augmented_y)[indexes]

    return np.append(X, augmented_X, axis=0), np.append(y, augmented_y)

def convert_to_rgb(images):
    """
    Getting back RGB images from transformed vectors given in this challenge
    """
    # Assuming images is an array of shape (n_images, n_pixels)
    n_images = images.shape[0]
    rgb_images = np.zeros((n_images, 32, 32, 3), dtype=np.uint8)  # Initialize array to store RGB images

    for i in range(n_images):
        # Separate the channels
        red_channel = images[i, :1024].reshape(32, 32)
        green_channel = images[i, 1024:2048].reshape(32, 32)
        blue_channel = images[i, 2048:].reshape(32, 32)

        # Find the minimum and maximum values for each channel
        min_red, max_red = np.min(red_channel), np.max(red_channel)
        min_green, max_green = np.min(green_channel), np.max(green_channel)
        min_blue, max_blue = np.min(blue_channel), np.max(blue_channel)
        min_, max_= min([min_red,min_green,min_blue]),max([max_red,max_green,max_blue])

        # Rescale each channel to range [0, 255]
        red_channel_rescaled = ((red_channel - min_red) / (max_red - min_red) * 255).astype(np.uint8)
        green_channel_rescaled = ((green_channel - min_green) / (max_green - min_green) * 255).astype(np.uint8)
        blue_channel_rescaled = ((blue_channel - min_blue) / (max_blue - min_blue) * 255).astype(np.uint8)

        # Combine channels to form the RGB image
        rgb_images[i] = np.stack([red_channel_rescaled, green_channel_rescaled, blue_channel_rescaled], axis=2)

    return rgb_images

def reshape(images):
    """
    Reshaping images from unidimensional vectors into RGB format
    """
    # Assuming images is an array of shape (n_images, n_pixels)
    n_images = images.shape[0]
    rgb_images = np.zeros((n_images, 32, 32, 3))  # Initialize array to store RGB images

    for i in range(n_images):
        # Reshape the image into the RGB format
        rgb_images[i, :, :, 0] = images[i, :1024].reshape(32, 32)  # Red channel
        rgb_images[i, :, :, 1] = images[i, 1024:2048].reshape(32, 32)  # Green channel
        rgb_images[i, :, :, 2] = images[i, 2048:].reshape(32, 32)  # Blue channel

    return rgb_images