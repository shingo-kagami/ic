import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

import ic_utils as ic


@jit
def generate_grad_orientation_histogram(mag, angle, n_bins=32):
    height = mag.shape[0]
    width = mag.shape[1]
    hist = np.zeros(n_bins, dtype=np.float32)

    for j in range(height):
        for i in range(width):
            m = mag[j, i]
            a = angle[j, i]
            if a == 2 * math.pi:
                ## prevent at_bin from getting out of [0, n_bins-1] range
                a = 0.0
            at_bin = int(a * n_bins / (2 * math.pi))
            hist[at_bin] += m

    return hist


def main():
    cap = ic.select_capture_source(sys.argv)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x) + math.pi  ## (0, 2*pi]

        n_bins = 32
        hist = generate_grad_orientation_histogram(mag, angle, n_bins)
        ax.clear()
        ax.bar(np.arange(n_bins) * 360 / n_bins, hist,
               width=1.01 * 360 / n_bins)
        plt.pause(0.001)

        vis_grad = np.empty((img.shape[0], img.shape[1], 3), np.float32)
        vis_grad[:, :, 0] = angle * 360 / (2 * math.pi)  ## hue in [0, 360]
        vis_grad[:, :, 1] = 1.0  ## sat in [0, 1]
        vis_grad[:, :, 2] = mag / mag.max()  ## val in [0, 1]
        vis_grad = cv2.cvtColor(vis_grad, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame', frame)
        cv2.imshow('gradient orientation', vis_grad)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
