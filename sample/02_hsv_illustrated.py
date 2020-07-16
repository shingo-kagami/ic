import math
import cv2
import numpy as np
from numba import jit

@jit
def generateHsvChart(dispsize, radius, val):
    hsv_chart = np.zeros((dispsize, dispsize, 3), np.float32)
    center = dispsize / 2
    for j in range(dispsize):
        for i in range(dispsize):
            px = i - center
            py = -(j - center)
            hue = math.atan2(py, px) * 180.0 / math.pi
            if hue < 0.0:
                hue += 360.0
            sat = math.sqrt(px * px + py * py) / radius if radius > 0 else 0
            if sat < 1.0:
                hsv_chart[j, i] = [hue * 0.5, sat * 255, val * 255]
            else:
                hsv_chart[j, i] = [0, 0, 0]
    return hsv_chart

def doNothing(x):
    pass

if __name__ == '__main__':

    cv2.namedWindow('HSV')
    cv2.createTrackbar('Value', 'HSV', 200, 255, doNothing)

    dispsize = 512
    max_radius = 250

    while True:

        val = cv2.getTrackbarPos('Value', 'HSV') / 255.0
        radius = max_radius * val
        hsv_chart = generateHsvChart(dispsize, radius, val)
        hsv_in_bgr = cv2.cvtColor(np.uint8(hsv_chart), cv2.COLOR_HSV2BGR)

        cv2.imshow("HSV", hsv_in_bgr)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
