import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

import ic_utils as ic


@jit
def generate_pixel_value_histogram(src, n_bins=256):
    height = src.shape[0]
    width = src.shape[1]

    hist_val = np.zeros(n_bins)
    for j in range(height):
        for i in range(width):
            at_bin = int(src[j, i] * n_bins / 256)
            hist_val[at_bin] += 1
    return hist_val


def main():
    cv2.namedWindow('result')
    cv2.createTrackbar('vmin', 'result', 0, 255, ic.do_nothing)
    cv2.createTrackbar('vmax', 'result', 255, 255, ic.do_nothing)

    cap = ic.select_capture_source(sys.argv)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        vmin = cv2.getTrackbarPos('vmin', 'result')
        vmax = cv2.getTrackbarPos('vmax', 'result')
        if vmax <= vmin:
            vmax = vmin + 1  ## make sure that vmax > vmin

        #n_bins = 256
        #n_bins = 128
        n_bins = 64
        pix_val_hist = generate_pixel_value_histogram(img, n_bins)

        ax.clear()
        ax.bar(np.int32(np.arange(n_bins) * 256 / n_bins),
               pix_val_hist, width=1.01 * (256 / n_bins))
        ax.hlines([0], vmin, vmax, 'red', linestyle='solid', linewidth=4)
        plt.pause(0.001)

        converted_img = np.uint8(
            np.clip((255 * (np.int32(img) - vmin)) / (vmax - vmin), 0, 255)
        )

        cv2.imshow('result', converted_img)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
