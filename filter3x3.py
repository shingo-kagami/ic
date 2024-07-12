import sys
import cv2
import numpy as np
from numba import jit

import ic_utils as ic


@jit
def filter3x3_impl(src, weight):
    """
    src, weigth: np.array(dtype=np.float32)
    """
    dest = np.zeros_like(src)
    width = src.shape[1]
    height = src.shape[0]

    for j in range(1, height - 1):
        for i in range(1, width - 1):
            sum = 0.0
            for n in range(3):
                for m in range(3):
                    sum += weight[n, m] * src[j + n - 1, i + m - 1]
            dest[j, i] = sum

    return dest


def get_weight(filter_type):
    if filter_type == 0:
        is_signed = False
        weight = 1.0 / 8 * np.array([[0, 1, 0],
                                     [1, 4, 1], 
                                     [0, 1, 0]])
    elif filter_type == 1:
        is_signed = True        
        weight = 1.0 / 8 * np.array([[-1, 0, 1],
                                     [-2, 0, 2], 
                                     [-1, 0, 1]])
    elif filter_type == 2:
        is_signed = True        
        weight = 1.0 / 8 * np.array([[-1, -2, -1],
                                     [0, 0, 0], 
                                     [1, 2, 1]])
    elif filter_type == 3:
        is_signed = True        
        weight = 1.0 / 1 * np.array([[0, 1, 0],
                                     [1, -4, 1], 
                                     [0, 1, 0]])
    elif filter_type == 4:
        is_signed = False
        weight = 1.0 / 1 * np.array([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]])
    else:
        is_signed = False
        weight = 1.0 / 1 * np.array([[0, 0, 0],
                                     [0, 1, 0], 
                                     [0, 0, 0]])

    return weight, is_signed


def main():
    cv2.namedWindow('result')
    cv2.createTrackbar('type', 'result', 0, 5, ic.do_nothing)
    cv2.createTrackbar('repeat', 'result', 1, 20, ic.do_nothing)
    cv2.createTrackbar('my; cv', 'result', 0, 1, ic.do_nothing)

    cap = ic.select_capture_source(sys.argv)

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = np.float32(frame) / 255.0

        impl = cv2.getTrackbarPos('my; cv', 'result')
        filter_type = cv2.getTrackbarPos('type', 'result')
        weight, is_signed = get_weight(filter_type)
        n_repeats = cv2.getTrackbarPos('repeat', 'result')

        for iter in range(n_repeats):
            if impl == 0:
                image = filter3x3_impl(image, weight)
            else:
                image = cv2.filter2D(image, cv2.CV_32F, weight)

        cv2.imshow("original", frame)
        if is_signed:
            cv2.imshow("result", image + 0.5)
        else:
            cv2.imshow("result", image)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
