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

@jit
def warpPerspective_impl(src, H, dest_shape):
    width = dest_shape[1]
    height = dest_shape[0]
    swidth = src.shape[1]
    sheight = src.shape[0]
    dest = np.empty((dest_shape[0], dest_shape[1], 3), dtype=np.uint8)
    Hinv = np.linalg.inv(H)
    for j in range(height):
        x0 = Hinv[0, 1] * j + Hinv[0, 2]
        y0 = Hinv[1, 1] * j + Hinv[1, 2]
        w0 = Hinv[2, 1] * j + Hinv[2, 2]
        for i in range(width):
            x = Hinv[0, 0] * i + x0;
            y = Hinv[1, 0] * i + y0;
            w = Hinv[2, 0] * i + w0;
            winv = 1.0 / w if w != 0.0 else 0.0
            x *= winv
            y *= winv
            x = min(swidth - 2.0, max(0.0, x))
            y = min(sheight - 2.0, max(0.0, y))
            x_int = int(x)
            x_frac = x - x_int
            y_int = int(y)
            y_frac = y - y_int
            weight = np.array([(1 - x_frac) * (1 - y_frac), 
                               x_frac * (1 - y_frac), 
                               (1 - x_frac) * y_frac, 
                               x_frac * y_frac])
            dest[j, i] = (weight[0] * src[y_int, x_int] 
                          + weight[1] * src[y_int, x_int + 1]
                          + weight[2] * src[y_int + 1, x_int]
                          + weight[3] * src[y_int + 1, x_int + 1])
    return dest

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
        #output = cv2.warpPerspective(input, H, (dest_size, dest_size))
        output = warpPerspective_impl(input, H, (dest_size, dest_size))

        disp = input.copy()
        for k in range(4):
            ## OpenCV 4 requires integer coordinates for center of circle
            cv2.circle(disp, src_pnts[k].astype(np.int32), 3, (0, 255, 0), -1)

        cv2.polylines(disp, [np.int32(src_pnts)], True, (0, 255, 0), 1)

        cv2.imshow("input", disp)
        cv2.imshow("output", output)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
