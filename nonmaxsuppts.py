# Author: TK
from scipy.ndimage import filters, interpolation
import numpy as np
import cv2
#import matplotlib.pyplot as plt


def nonmaxsuppts(cim,radius,threshold):
    # Extract local maxima by performing a grey scale morphological
    # dilation and then finding points in the corner strength image that
    # match the dilated image and are also greater than the threshold.
    # non-maximum suppression in radius x radius regions

    h, w = cim.shape[0:2]
    cap_length = 1000

    maxH = filters.maximum_filter(cim, (radius,radius))

    cim = cim * (cim == maxH)
    cim = cim * (cim > threshold)
    

    # sort points by strength and find their positions
    sortIdx = np.argsort(cim.flatten())[::-1]

    count = 0
    for row in range(0,cim.shape[0]):
        for col in range(0,cim.shape[1]):
            if cim[row,col]:
                count += 1

    sortIdx = sortIdx[0:min(count,cap_length)]
    yy = sortIdx / w
    xx = sortIdx % w

    ret_list = list()
    for i in range(0,len(sortIdx)):
        ret_list.append((yy[i],xx[i]))
    return ret_list # THIS IS OF TYPE   list<tuples>