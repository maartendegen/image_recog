from analysis.lib.image_analysis import camera_tools as ct
import numpy as np


def cam_load_image(filepath = None):
    if filepath == None:
        filepath = '/Users/maartendegen/Documents/measuring/analysis/notebooks/basic/051319_scan2d_Hillary_NVsearch_scan5_focus=52.2um_zrel=1.5_um'
        
    return np.loadtxt('latest_image.dat', delimiter = '\t', ndmin = 2)

def stamp_out_relevant_field_of_view(array2d, xsize = 300, ysize = 300, xoffset = 10, yoffset = 60):
    
    (xlen, ylen) = np.shape(array2d)
    
    xmin = int(xlen/2.-xsize/2.)+xoffset
    xmax = int(xlen/2.+xsize/2.)+xoffset
    ymin = int(ylen/2.-ysize/2.)+yoffset
    ymax = int(ylen/2.+ysize/2.)+yoffset
    
    return array2d[xmin:xmax, ymin:ymax]

testfoto = stamp_out_relevant_field_of_view(cam_load_image())

testfoto = ct.apply_brightness_filter(testfoto, 0)

fig, ax = plt.subplots()
ax.imshow(testfoto, interpolation='nearest', cmap=plt.cm.gray)

plt.show()
