import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spi
import scipy.ndimage as ndimage
    
%matplotlib inline

fig, (ax1) = plt.subplots(1, figsize=(10, 10))
fig, (ax2) = plt.subplots(1, figsize=(10, 10))

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

image = gaussian_laplace(image)
image = sobel(image)
image = max_filter(image,50)

ax2.imshow(image)