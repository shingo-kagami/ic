import math
import cv2
import numpy as np
from numba import jit

import ic_utils as ic


def getPerspectiveTransform_impl(src_pnts, dest_pnts):
    """
    Given:
      (x_k, y_k): source points (k = 0, 1, 2, 3)
      (u_k, v_k): destination points (k = 0, 1, 2, 3)

    Solve:
        [u_k]   [ h11 h12 h13 ] [x_k]
    s_k [v_k] = [ h21 h22 h23 ] [y_k]
        [ 1 ]   [ h31 h32  1  ] [ 1 ]

    to obtain 8 unknown variables (while h33 is fixed to 1):
      h11, h12, ..., h32

    Note that s_k (k = 0, 1, 2, 3) are also unknown.

    ----
    Since the 3rd row means s_k = h31 x_k + h32 y_k + 1, the 1st and
    2nd rows yield

      h31 x_k u_k + h32 y_k u_k + u_k = h11 x_k h12 y_k + h13
      h31 x_k v_k + h32 y_k v_k + v_k = h21 x_k h22 y_k + h23

    or equivalently, 

    [x_k y_k  1   0   0   0  -x_k u_k  -y_k u_k] [h11] = [u_k]
    [ 0   0   0  x_k y_k  1  -x_k v_k  -y_k v_k] [h12]   [v_k]
                                                 [h13]
                                                 [h21]
                                                 [h22]
                                                 [h23]
                                                 [h31]
                                                 [h32]

    By stacking them for k = 0, 1, 2, 3, we have 8 equations with 8
    unknowns ready to be solved.
    """

    x = src_pnts[:, 0]
    y = src_pnts[:, 1]
    u = dest_pnts[:, 0]
    v = dest_pnts[:, 1]

    list_A = []
    list_b = []
    for k in range(4):
        list_A.append([x[k], y[k], 1, 0, 0, 0, -x[k]*u[k], -y[k]*u[k]])
        list_A.append([0, 0, 0, x[k], y[k], 1, -x[k]*v[k], -y[k]*v[k]])
        list_b.append([u[k]])
        list_b.append([v[k]])

    A = np.array(list_A, dtype=np.float64)
    b = np.array(list_b, dtype=np.float64)
    h = np.linalg.solve(A, b)  ## solve A h = b
    h = np.vstack([h, np.array([1.0])])  ## append the last element h33 (= 1.0)

    return h.reshape((3, 3))


@jit
def warpPerspective_impl(src, H, dest_shape):
    height = dest_shape[0]
    width = dest_shape[1]
    sheight = src.shape[0]
    swidth = src.shape[1]
    dest = np.empty((dest_shape[0], dest_shape[1], 3), dtype=np.uint8)
    Hinv = np.linalg.inv(H)

    for j in range(height):
        x0 = Hinv[0, 1] * j + Hinv[0, 2]
        y0 = Hinv[1, 1] * j + Hinv[1, 2]
        w0 = Hinv[2, 1] * j + Hinv[2, 2]
        for i in range(width):
            x = Hinv[0, 0] * i + x0
            y = Hinv[1, 0] * i + y0
            w = Hinv[2, 0] * i + w0

            if w != 0.0:
                x, y = x / w, y / w
            else:
                x, y = 0.0, 0.0

            x = min(swidth - 1.0, max(0.0, x))
            y = min(sheight - 1.0, max(0.0, y))

            ## integer and fractional parts of coordinates
            x_int, y_int = int(x), int(y)
            x_frac, y_frac = x - x_int, y - y_int

            ## weighted sum of four neighboring pixels at integer coordinates
            weight = np.array([(1 - x_frac) * (1 - y_frac), 
                               x_frac * (1 - y_frac), 
                               (1 - x_frac) * y_frac, 
                               x_frac * y_frac])
            dest[j, i] = (weight[0] * src[y_int, x_int] 
                          + weight[1] * src[y_int, x_int + 1]
                          + weight[2] * src[y_int + 1, x_int]
                          + weight[3] * src[y_int + 1, x_int + 1])

    return dest


def on_mouse(event, x, y, flag, mstate):
    if event == cv2.EVENT_LBUTTONDOWN:
        n_points = len(mstate['src_pnts'])
        for k in range(n_points):
            cx, cy = mstate['src_pnts'][k]
            distance_squared = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            if distance_squared < 10:
                mstate['pnt_selected'] = k
                break

    if event == cv2.EVENT_LBUTTONUP:
        mstate['pnt_selected'] = -1

    if mstate['pnt_selected'] >= 0:
        k = mstate['pnt_selected']
        mstate['src_pnts'][k] = np.array([x, y])


def main():
    frame = cv2.imread('lena.jpg')

    cv2.namedWindow('source')
    cv2.createTrackbar('my; cv', 'source', 0, 1, ic.do_nothing)

    mstate = {
        'src_pnts': np.array([[100, 100],
                              [100, 200],
                              [200, 200],
                              [200, 100]],
                             dtype=np.float32),
        'pnt_selected': -1,
    }
    cv2.setMouseCallback('source', on_mouse, mstate)

    dest_size = (256, 256)
    dest_pnts = np.array([[0, 0],
                          [0, dest_size[1]],
                          [dest_size[0], dest_size[1]],
                          [dest_size[0], 0]],
                         dtype=np.float32)

    while True:
        impl = cv2.getTrackbarPos('my; cv', 'source')

        src_pnts = mstate['src_pnts']

        if impl == 0:
            try:
                H = getPerspectiveTransform_impl(src_pnts, dest_pnts)
                result = warpPerspective_impl(frame, H, dest_size)
            except np.linalg.linalg.LinAlgError:
                ## just skip if singular
                print("LinAlgError")
        else:
            H = cv2.getPerspectiveTransform(src_pnts, dest_pnts)
            result = cv2.warpPerspective(frame, H, dest_size)

        disp = frame.copy()
        for pt in src_pnts:
            cv2.circle(disp, np.int32(pt), 3, (0, 255, 0), -1)

        cv2.polylines(disp, [np.int32(src_pnts)], True, (0, 255, 0), 1)

        cv2.imshow('source', disp)
        cv2.imshow('destination', result)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
