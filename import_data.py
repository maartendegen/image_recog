import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage import feature

nr_of_scans = 1
folder = '/Users/maartendegen/Documents/measuring/analysis/notebooks/basic/051319_scan2d_Hillary_NVsearch_scan5_focus=52.2um_zrel=1.5_um'

print folder
d = ds.DisplayScan(folder)
x,y,cdata = d.get_data()
