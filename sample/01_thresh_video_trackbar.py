import cv2
import sys
import numpy as np
from numba import jit

@jit
def threshold_impl(src, thresh, maxval):
    width = src.shape[1]
    height = src.shape[0]
    dest = np.zeros_like(src)
    for j in range(height):
        for i in range(width):
            if src[j, i] > thresh:
                dest[j, i] = maxval
            else:
                dest[j, i] = 0
    return dest

def doNothing(x):
    pass

if __name__ == '__main__':
    cv2.namedWindow('test')
    cv2.createTrackbar('thresh', 'test', 128, 255, doNothing)

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
        thresh_val = cv2.getTrackbarPos('thresh', 'test')
        thresh_img = threshold_impl(input, thresh_val, 255)

        cv2.imshow("test", thresh_img)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
