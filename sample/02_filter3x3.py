import cv2
import sys
import numpy as np
from numba import jit

@jit
def saturate_uint8(x):
    return min(max(0, x), 255)

@jit
def saturate_int8(x):
    return min(max(-128, x), 127)

@jit
def filter3x3_impl(src, weight, signed=False):
    width = src.shape[1]
    height = src.shape[0]
    dest = np.zeros_like(src)

    for j in range(1, height - 1):
        for i in range(1, width - 1):
            sum = 0.0
            for n in range(3):
                for m in range(3):
                    sum += weight[n, m] * src[j + n - 1, i + m - 1]

            if signed == True:
                dest[j, i] = int(saturate_int8(sum))
            else:
                dest[j, i] = int(saturate_uint8(sum))
    return dest

def doNothing(x):
    pass

if __name__ == '__main__':
    cv2.namedWindow('filtered')
    cv2.createTrackbar('nIteration', 'filtered', 1, 20, doNothing)

    cap_src = 'vtest.avi'
    if len(sys.argv) == 2:
        if sys.argv[1].isdecimal():
            cap_src = int(sys.argv[1])
        else:
            cap_src = sys.argv[1]
    cap = cv2.VideoCapture(cap_src)

    weight1 = 1.0 / 8 * np.array([[0, 1, 0],
                                  [1, 4, 1], 
                                  [0, 1, 0]])

    weight2 = 1.0 / 8 * np.array([[-1, 0, 1],
                                  [-2, 0, 2], 
                                  [-1, 0, 1]])

    while True:
        retval, input = cap.read()
        if retval == False:
            break
        input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        image = input.copy()  # create a copy of the data, not a reference


        image = filter3x3_impl(image, weight1)
        #image = cv2.filter2D(image, cv2.CV_8U, weight1)

        # n = cv2.getTrackbarPos('nIteration', 'filtered')
        # for i in range(n):
        #     image = filter3x3_impl(image, weight1)

        #image = np.int8(filter3x3_impl(image, weight2, signed=True))
        #image = np.int8(cv2.filter2D(image, cv2.CV_32F, weight2))

        cv2.imshow("original", input)
        cv2.imshow("filtered", image)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
