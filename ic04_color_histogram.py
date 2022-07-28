import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from numba import jit


@jit
def generate_hue_sat_histogram(image, hist_size, is_weighted=True):
    width = image.shape[1]
    height = image.shape[0]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    w_normalizer = 2.0 / width
    h_normalizer = 2.0 / height

    hbin = hist_size[0]
    sbin = hist_size[1]
    hist = np.zeros((sbin, hbin))

    for j in range(height):
        for i in range(width):
            weight = 1.0
            if is_weighted:
                x_norm = (i - cx) * w_normalizer
                y_norm = (j - cy) * h_normalizer
                rr = x_norm ** 2 + y_norm ** 2
                if rr <= 1:
                    weight = 1 - rr
                else:
                    weight = 0.0

            hue = image[j, i, 0]
            sat = image[j, i, 1]
            hist[int(sat * sbin / 256.0), int(hue * hbin / 180.0)] += weight

    hist_sum = np.sum(hist)
    hist /= hist_sum
    return hist


def generate_hue_sat_legend(hbin, sbin):
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


def plot_hue_sat_histogram(ax, hist, legend):
    hbin = hist.shape[1]
    sbin = hist.shape[0]
    cols = np.arange(hbin)
    rows = np.arange(sbin)
    mesh_rows, mesh_cols = np.meshgrid(rows, cols)

    ## x and y are transposed so that the direction of y agrees with an image
    facecolors = cv2.cvtColor(legend.transpose(1, 0, 2), cv2.COLOR_BGR2RGB)
    surf = ax.plot_surface(mesh_rows, mesh_cols, hist.T, 
                           facecolors=facecolors/255.0)
    surf.set_edgecolor('black')
    surf.set_linewidth(0.2)
    ax.set_ylabel('Hue')
    ax.set_xlabel('Saturation')


def main():
    file = 'lena.jpg'
    if len(sys.argv) == 2:
        file = sys.argv[1]

    img = cv2.imread(file)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = generate_hue_sat_histogram(img_hsv, (64, 64), is_weighted=False)
    legend = generate_hue_sat_legend(64, 64)

    cv2.imshow('img', img)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(35, -35)
    plot_hue_sat_histogram(ax, hist, legend)
    plt.show()
    
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
