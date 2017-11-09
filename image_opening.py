import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
    
%matplotlib inline

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 20))

#Import  Read H5 file
f = h5.File("051319_scan2d_Hillary_NVsearch_scan5_focus=52.2um_zrel=1.5_um.hdf5", "r")
x, y, image = f['x'], f['y'], f['countrate']

ax1.imshow(image)
