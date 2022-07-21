import sys
import cv2
import numpy as np
from numba import jit

import ic_utils as ic


def min_eigen_value_map(H):
    # return np.linalg.eigvals(H).min(axis=2)  ## This is slow!

    a = H[:, :, 0, 0]  # H = [a b]
    b = H[:, :, 0, 1]  #     [c d]
    c = H[:, :, 1, 0]
    d = H[:, :, 1, 1]

    ## the smaller solution of s^2 - (a + d) s + ad - bc = 0
    min_eig = ((a + d) - np.sqrt((a - d)**2 + 4 * b * c)) / 2

    return min_eig


def harris_map(H, coeff_k):
    a = H[:, :, 0, 0]  # H = [a b]
    b = H[:, :, 0, 1]  #     [c d]
    c = H[:, :, 1, 0]
    d = H[:, :, 1, 1]

    return (a * d - b * c) - coeff_k * (a + d)**2
    
    
def hessian_map(T, block_size=5):
    Tx = np.gradient(T, axis=1)
    Ty = np.gradient(T, axis=0)
    TxTx = Tx * Tx
    TyTy = Ty * Ty
    TxTy = Tx * Ty

    theight = T.shape[0]
    twidth = T.shape[1]
    H = np.zeros((theight, twidth, 2, 2), dtype=T.dtype)
    H[:, :, 0, 0] = cv2.blur(TxTx, (block_size, block_size))
    H[:, :, 1, 1] = cv2.blur(TyTy, (block_size, block_size))
    H[:, :, 0, 1] = cv2.blur(TxTy, (block_size, block_size))
    H[:, :, 1, 0] = H[:, :, 0, 1]

    return H


@jit
def feature_point_list(img, threshold=1.0, suppress_nonmax=True):
    """
    Returns:
      [ [x_1, y_1, value_1],
        [x_2, y_2, value_2],
        ...              
        [x_k, y_k, value_k] ]
      in descending order of value's
    """
    height, width = img.shape
    result = []

    for j in range(1, height - 1):
        for i in range(1, width - 1):
            if img[j, i] < threshold:
                continue

            if not suppress_nonmax:
                result.append([i, j, img[j, i]])
                continue

            maxval = 0.0
            for n in range(3):
                for m in range(3):
                    val = img[j + n - 1, i + m - 1]
                    if val > maxval:
                        maxval = val

            if img[j, i] == maxval:
                result.append([i, j, maxval])

    return sorted(result, key=lambda x: x[2], reverse=True)


def main():
    cv2.namedWindow('result')
    cv2.createTrackbar('num points', 'result', 1000, 5000, ic.do_nothing)
    cv2.createTrackbar('mineig; harris', 'result', 0, 1, ic.do_nothing)
    cv2.createTrackbar('my; cv', 'result', 0, 1, ic.do_nothing)

    cap = ic.select_capture_source(sys.argv)
    harris_k = 0.04

    while True:
        grabbed, frame_color = cap.read()
        if not grabbed:
            break
        frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        T = np.float32(frame)

        map_type = cv2.getTrackbarPos('mineig; harris', 'result')
        n_points = cv2.getTrackbarPos('num points', 'result')
        impl = cv2.getTrackbarPos('my; cv', 'result')

        if map_type == 0:
            response_map = min_eigen_value_map(hessian_map(T))
        else:
            response_map = harris_map(hessian_map(T), harris_k)
            
        if impl == 0:
            fp_list = feature_point_list(response_map)
            fp_list = fp_list[0:n_points]
        else:
            fp_list = cv2.goodFeaturesToTrack(frame, n_points,
                                              qualityLevel=1e-10,
                                              minDistance=2.0,
                                              useHarrisDetector=map_type,
                                              k=harris_k)
            ## reshaping from (#points, 1, 2) to (#points, 2)
            fp_list = fp_list.reshape(-1, 2)

        for fp in fp_list:
            cv2.circle(frame_color, np.int16(fp[:2]), 2, (0, 255, 0), -1)

        cv2.imshow('response_map', response_map / response_map.max())
        cv2.imshow('result', frame_color)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
