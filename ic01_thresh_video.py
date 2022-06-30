import sys
import cv2
import numpy as np
from numba import jit

import ic_utils as ic


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


def main():
    cap = ic.select_capture_source(sys.argv)

    cv2.namedWindow('result')
    cv2.createTrackbar('thresh', 'result', 128, 255, ic.do_nothing)
    cv2.createTrackbar('my; cv; np', 'result', 0, 2, ic.do_nothing)

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        th = cv2.getTrackbarPos('thresh', 'result')
        impl = cv2.getTrackbarPos('my; cv; np', 'result')

        if impl == 0: ## my implementation
            thresh_img = threshold_impl(frame, th, 255)
        elif impl == 1: ## opencv implementation
            ret, thresh_img = cv2.threshold(frame, th, 255, cv2.THRESH_BINARY)
        elif impl == 2: ## numpy implementation
            thresh_img = np.where(frame > th, np.uint8(255), np.uint8(0))

        cv2.imshow('result', thresh_img)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
