import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from numba import jit

@jit
def createHueSatHistogram(image, hist_size, is_weighted = True):
    width = image.shape[1]
    height = image.shape[0]
    w_normalizer = 2.0 / width
    h_normalizer = 2.0 / height
    cx = int(width / 2)
    cy = int(height / 2)

    hbin = hist_size[0]
    sbin = hist_size[1]
    hist = np.zeros((sbin, hbin))

    for j in range(height):
        for i in range(width):
            weight = 1.0
            if is_weighted:
                x_norm = (i - cx) * w_normalizer
                y_norm = (j - cy) * h_normalizer
                rr = x_norm * x_norm + y_norm * y_norm
                weight = 1 - rr if (rr <= 1) else 0.0
            hue = image[j, i, 0]
            sat = image[j, i, 1]
            hist[int(sat * sbin / 256.0), int(hue * hbin / 180.0)] += weight

    hist_sum = np.sum(hist)
    hist /= hist_sum
    return hist


def createHueSatLegend(hbin, sbin):
    width = hbin
    height = sbin
    disp = np.zeros((height, width, 3), np.uint8)

    for j in range(height):
        for i in range(width):
            hue = (i * 180.0) / width
            sat = (j * 256.0) / height
            val = 255
            disp[j, i] = (int(hue), int(sat), val)
    return cv2.cvtColor(disp, cv2.COLOR_HSV2BGR)


def plotHueSatHistogram(ax, hist, legend):
    hbin = hist.shape[1]
    sbin = hist.shape[0]
    cols = np.arange(hbin)
    rows = np.arange(sbin)
    mesh_rows, mesh_cols = np.meshgrid(rows, cols)

    # x and y are exchanged so that the direction of y agrees with an image
    facecolors = cv2.cvtColor(legend.transpose(1, 0, 2), cv2.COLOR_BGR2RGB)
    surf = ax.plot_surface(mesh_rows, mesh_cols, hist.T, 
                           facecolors=facecolors/255.0)
    surf.set_edgecolor("black")
    surf.set_linewidth(0.2)
    ax.set_ylabel('Hue')
    ax.set_xlabel('Saturation')
