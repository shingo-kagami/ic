import cv2
import numpy as np

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

if __name__ == '__main__':
    input = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

    retval, thresh_cv = cv2.threshold(input, 128, 255, cv2.THRESH_BINARY)
    thresh_np = np.full_like(input, 255) * (input > 128)
    thresh_my = threshold_impl(input, 128, 255)

    cv2.imshow("test1", thresh_cv)
    cv2.imshow("test2", thresh_np)
    cv2.imshow("test3", thresh_my)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
