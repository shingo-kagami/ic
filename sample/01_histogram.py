import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit
def calcHistogram(src):
    width = src.shape[1]
    height = src.shape[0]
    pix_val_hist = np.zeros(256)
    for j in range(height):
        for i in range(width):
            pix_val_hist[src[j, i]] += 1
    return pix_val_hist

def doNothing(x):
    pass

if __name__ == '__main__':
    cv2.namedWindow('test')
    cv2.createTrackbar('vmin', 'test', 0, 255, doNothing)
    cv2.createTrackbar('vmax', 'test', 255, 255, doNothing)

    cap_src = 'vtest.avi'
    if len(sys.argv) == 2:
        if sys.argv[1].isdecimal():
            cap_src = int(sys.argv[1])
        else:
            cap_src = sys.argv[1]
    cap = cv2.VideoCapture(cap_src)

    while True:
        retval, input = cap.read()
        if retval == False:
            break
        input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

        vmin = thresh_val = cv2.getTrackbarPos('vmin', 'test')
        vmax = thresh_val = cv2.getTrackbarPos('vmax', 'test')
        if vmax <= vmin:
            vmax = vmin + 1

        plt.gcf().clear()
        pix_val_hist = calcHistogram(input)
        plt.bar(range(256), pix_val_hist)
        plt.hlines([0], vmin, vmax, "red", linestyle='solid', linewidth=4)
        plt.pause(0.01)

        converted_img = np.uint8(np.clip(
            (255 * (np.int32(input) - vmin)) / (vmax - vmin), 
            0, 255)
        )

        cv2.imshow("test", converted_img)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
