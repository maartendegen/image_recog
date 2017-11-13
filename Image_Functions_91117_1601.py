import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spi
import scipy.ndimage as ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops


#Import  Read H5 file
f = h5.File("051319_scan2d_Hillary_NVsearch_scan5_focus=52.2um_zrel=1.5_um.hdf5", "r")
x, y, image = f['x'], f['y'], f['countrate']

#image = np.array(image, dtype = np.uint16)
image = np.array(image)

def gaussian_laplace(image):
    image =  image - ndimage.gaussian_laplace(image, sigma=3)
    return image

def sobel(image):
    sx = ndimage.sobel(image, axis=0)  # gradient in x direction
    sy = ndimage.sobel(image, axis=1)  # gradient in y direction
    sob = np.sqrt(sx**2 + sy**2)  # square magnitude of gradient
    return sob

def max_filter(image,size=5):
    filtered_image = ndimage.maximum_filter(image,size)
    return filtered_image

def binarize(image):
    """
    Input : image (2d array)
    Output: Masked image where all values below the mean of the image are set to 0
    """
    
    #Calculate the mean of the image and mask the array to remove all values below
    image_mean = np.mean(image)
    
    #Make an output array with the same shape than the input image
    output = np.zeros_like(image)
    
    #Make a mask 
    mask = np.ma.masked_array(np.ma.masked_greater(image, image_mean)).mask
    
    #Convert to integer 
    output = np.array(mask, dtype = int)

    return output

def label_image(image):
    """
    This function labels an input image and returns the labeled array, the number of features, and the centers
    """
    
    #Label the blobs using ndimage
    labeled_blobs, n_features = ndimage.label(b_image)
    
    #calculate the center of mass of each labelled feature
    centers = ndimage.center_of_mass(b_image, labeled_blobs, np.arange(n_features) + 1)
    
    return labeled_blobs, n_features, centers
    
    
def determine_blob_size(labeled_blobs):
    """
    This function takes the labeled image array as input and returns the size of the features
    """
    properties = regionprops(labeled_blobs)
    
    size = []
    for p in properties:
        min_row, min_col, max_row, max_col = p.bbox
        size.append((float(max(max_row - min_row, max_col - min_col))))
        
    return size
    
b_image = binarize(image)
labeled_image = label_image(b_image)
blob_size = determine_blob_size(labeled_image[0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))

ax1.imshow(b_image, cmap='gray')
ax1.set_title('blobs', size=20)


ax2.imshow(labeled_image[0], cmap='viridis')
ax2.set_title('labelled blobs', size=20)


# mark the centers of mass on the image
x, y = zip(*labeled_image[2])  # unzip the pairs (x, y) into two lists
#ax2.scatter(y, x)  # invert the coordinates, as we are talking about row/column indices now



