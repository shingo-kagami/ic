import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from numba import jit

# Make sure you have "color_histogram_utils.py" in the same directory
from color_histogram_utils import createHueSatHistogram, createHueSatLegend, plotHueSatHistogram

if __name__ == '__main__':
    file = "lena.jpg"
    if len(sys.argv) == 2:
        file = sys.argv[1]

    img = cv2.imread(file)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = createHueSatHistogram(img_hsv, (64, 64), is_weighted=False)
    legend = createHueSatLegend(64, 64)

    cv2.imshow("img", img)
    cv2.imshow("hist", hist * 100.0)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(35, -35)
    plotHueSatHistogram(ax, hist, legend)
    plt.show()
    
    cv2.waitKey(0)
