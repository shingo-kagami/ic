import sys
import cv2
import numpy as np
from numba import jit

import ic_utils as ic


@jit
def convert_graylevels(src, vmin, vmax):
    width = src.shape[1]
    height = src.shape[0]
    dest = np.zeros_like(src)

    for j in range(height):
        for i in range(width):
            val = (255 * (src[j, i] - vmin)) / (vmax - vmin)
            val = max(0, val)   ## clipping negative values
            val = min(val, 255) ## clipping values over 255
            dest[j, i] = val

    return dest
    

def main():
    cv2.namedWindow('result')
    cv2.createTrackbar('vmin', 'result', 0, 255, ic.do_nothing)
    cv2.createTrackbar('vmax', 'result', 255, 255, ic.do_nothing)
    cv2.createTrackbar('my; np', 'result', 0, 1, ic.do_nothing)

    cap = ic.select_capture_source(sys.argv)

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        vmin = cv2.getTrackbarPos('vmin', 'result')
        vmax = cv2.getTrackbarPos('vmax', 'result')
        if vmax <= vmin:
            vmax = vmin + 1

        impl = cv2.getTrackbarPos('my; np', 'result')
        if impl == 0: ## my impl
            converted_img = convert_graylevels(img, vmin, vmax)
        else: ## numpy impl
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
