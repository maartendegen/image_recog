

import numpy as np
from scipy.ndimage import generate_binary_structure
import scipy.ndimage


def remove_small_values(image):
    """
    Input : image (2d array)
    Output: Masked image where all values below the mean of the image are set to 0
    """
    image_mean = np.mean(image)
    image = np.ma.masked_array(np.ma.masked_less(image, image_mean))
    return image


def labeler(image, do_plot = False):
    """
    A function that labels all connected features.
    Any non-zero elements are considered features
    and zero elements are considered background. 
    Input:
        image (np.ndarray): the input image
        
    Output:
        x (np.ndarray): the x-values of the center of the
            labeled features
        y (np.ndarray): the y-values of the center of the
            labeled features        
    """
    s = generate_binary_structure(2,2)

    labeled_blobs, n_features = ndimage.label(image, structure = s)
    centers = ndimage.center_of_mass(image, labeled_blobs, np.arange(n_features) + 1)
    x, y = zip(*centers)

    if do_plot:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))

        ax1.imshow(image)

        ax2.imshow(labeled_blobs, cmap='viridis')
        ax2.set_title('labelled blobs', size=20)


        # calculate the center of mass of each labelled feature
        centers = ndimage.center_of_mass(image, labeled_blobs, np.arange(n_features) + 1)

        # mark the centers of mass on the image
        x, y = zip(*centers)  # unzip the pairs (x, y) into two lists
        ax2.scatter(y, x)  # invert the coordinates, as we are talking about row/column indices now

    return x, y

def size_features(image):
    """
    A function that outputs an array with pixel sizes of labeled features.
    Input:
        image (np.ndarray): an image array with boolean values.
    Output:
        sizes (np.ndarray): an array with the sizes of labeled features
        (in units of pixels)
    """

    s = generate_binary_structure(2,2)

    labeled_blobs, n_features = ndimage.label(image, structure = s) 

    sizes = scipy.ndimage.measurements.sum(image, labeled_blobs, index = range(n_features))

    return sizes


