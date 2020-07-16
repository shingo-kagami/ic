import math
import cv2
import numpy as np
from numba import jit

pnt_selected = -1

src_pnts = np.array([[100, 100], 
                     [200, 100], 
                     [200, 200], 
                     [100, 200]], 
                    np.float32)

dest_size = 256
dest_pnts = np.array([[0, 0], 
                      [dest_size - 1, 0], 
                      [dest_size - 1, dest_size - 1], 
                      [0, dest_size - 1]], 
                     np.float32)

def onMouse(event, x, y, flag, param):
    global pnt_selected, src_pnts

    if event == cv2.EVENT_LBUTTONDOWN:
        for k in range(4):
            cx = src_pnts[k, 0]
            cy = src_pnts[k, 1]
            distance_squared = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            if distance_squared < 10:
                pnt_selected = k
                break

    if event == cv2.EVENT_LBUTTONUP:
        pnt_selected = -1

    if pnt_selected >= 0:
        src_pnts[pnt_selected] = np.array([x, y])


if __name__ == '__main__':
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', onMouse, None)

    input = cv2.imread('lena.jpg')

    while True:

        H = cv2.getPerspectiveTransform(src_pnts, dest_pnts)
        output = cv2.warpPerspective(input, H, (dest_size, dest_size))

        disp = input.copy()
        for k in range(4):
            cv2.circle(disp, tuple(src_pnts[k]), 3, (0, 255, 0), -1)

        cv2.polylines(disp, [np.int32(src_pnts)], True, (0, 255, 0), 1)

        cv2.imshow("input", disp)
        cv2.imshow("output", output)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
