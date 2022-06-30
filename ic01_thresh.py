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


def main():
    frame = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

    th = 128
    thresh_my = threshold_impl(frame, th, 255)
    ret, thresh_cv = cv2.threshold(frame, th, 255, cv2.THRESH_BINARY)
    thresh_np = np.where(frame > th, np.uint8(255), np.uint8(0))

    cv2.imshow('result', thresh_my)
    cv2.imshow('result2', thresh_cv)
    cv2.imshow('result3', thresh_np)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
