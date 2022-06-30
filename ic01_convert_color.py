import sys
import cv2
import numpy as np
from numba import jit

import ic_utils as ic

@jit
def cvtBGR2HSV_impl(src):
    width = src.shape[1]
    height = src.shape[0]
    dest = np.zeros_like(src)

    for j in range(height):
        for i in range(width):

            [b, g, r] = src[j, i] / 255.0
            max_rgb, min_rgb = max(r, g, b), min(r, g, b)
            diff = max_rgb - min_rgb

            if max_rgb == min_rgb: 
                hue = 0
            elif max_rgb == r: 
                hue = 60 * ((g - b) / diff)     
            elif max_rgb == g: 
                hue = 60 * ((b - r) / diff) + 120  
            elif max_rgb == b: 
                hue = 60 * ((r - g) / diff) + 240

            if hue < 0: 
                hue += 360
            
            if max_rgb != 0: 
                sat = diff / max_rgb
            else: 
                sat = 0

            val = max_rgb

            dest[j, i] = [hue * 0.5, sat * 255, val * 255]

    return dest


@jit
def cvtHSV2BGR_impl(src):
    width = src.shape[1]
    height = src.shape[0]
    dest = np.zeros_like(src)

    for j in range(height):
        for i in range(width):
            h_int, s_int, v_int = src[j, i]
            h = h_int / 180.0
            s = s_int / 255.0
            v = v_int / 255.0
            r = v
            g = v
            b = v
            if s > 0.0:
                h *= 6.0
                h_int = int(h)
                h_frac = h - 1.0 * h_int
                if h_int == 0:
                    g *= 1 - s * (1 - h_frac)
                    b *= 1 - s
                elif h_int == 1:
                    r *= 1 - s * h_frac
                    b *= 1 - s
                elif h_int == 2:
                    r *= 1 - s
                    b *= 1 - s * (1 - h_frac)
                elif h_int == 3:
                    r *= 1 - s
                    g *= 1 - s * h_frac
                elif h_int == 4:
                    r *= 1 - s * (1 - h_frac)
                    g *= 1 - s
                elif h_int == 5:
                    g *= 1 - s
                    b *= 1 - s * h_frac
            dest[j, i] = [255 * b, 255 * g, 255 * r]

    return dest


@jit
def hue_image(src, s_thresh=64, v_thresh=64):
    width = src.shape[1]
    height = src.shape[0]
    dest = np.zeros_like(src)

    for j in range(height):
        for i in range(width):
            h_int, s_int, v_int = src[j, i]
            if s_int < s_thresh:
                dest[j, i] = [0, 0, 0]
            elif v_int < v_thresh:
                dest[j, i] = [0, 0, 0]
            else:
                dest[j, i] = [h_int, 255, 255]
    return dest


if __name__ == '__main__':
    cap = ic.select_capture_source(sys.argv)

    cv2.namedWindow('frame')
    cv2.namedWindow('result')
    cv2.createTrackbar('my; cv', 'result', 0, 1, ic.do_nothing)

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        impl = cv2.getTrackbarPos('my; cv', 'result')
        if impl == 0: ## my impl
            hsv_img = cvtBGR2HSV_impl(frame)
            hsv_img = hue_image(hsv_img)
            result = cvtHSV2BGR_impl(hsv_img)
        else: ## opencv impl
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_img = hue_image(hsv_img)
            result = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame', frame)
        cv2.imshow('result', result)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
