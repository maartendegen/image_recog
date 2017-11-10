import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spi
import scipy.ndimage as ndimage
   
fig, (ax1,ax2,ax3) = plt.subplots(3, figsize=(10, 10))

#Import  Read H5 file
f = h5.File("051319_scan2d_Hillary_NVsearch_scan5_focus=52.2um_zrel=1.5_um.hdf5", "r")
x, y, image = f['x'], f['y'], f['countrate']
ax1.imshow(image)


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

def remove_small_values(image):
    """
    Input : image (2d array)
    Output: Masked image where all values below the mean of the image are set to 0
    """
    image_mean = np.mean(image)
    image = np.ma.masked_array(np.ma.masked_less(image, image_mean))
    return image

max_filter = max_filter(image,20)
ax2.imshow(max_filter)

markers = remove_small_values(max_filter)
ax3.imshow(markers)


# labeled_markers, n_features = ndimage.label(image)

# ax2.imshow(labeled_markers, cmap='viridis')
# ax2.set_title('labelled markers', size=20)

# # calculate the center of mass of each labelled feature
# centers = ndimage.center_of_mass(markers, labeled_markers, np.arange(n_features) + 1)
# # mark the centers of mass on the image
# x, y = zip(*centers)  # unzip the pairs (x, y) into two lists
# ax3.scatter(y, x)  # invert the coordinates, as we are talking about row/column indices now

plt.show()
