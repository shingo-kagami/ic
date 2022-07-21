import sys
import cv2
import numpy as np
from numpy.linalg import inv

import ic_utils as ic


def compute_derivatives(T):
    theight = T.shape[0]
    twidth = T.shape[1]
    npix = twidth * theight

    Tx = np.gradient(T, axis=1).reshape(npix, 1)
    Ty = np.gradient(T, axis=0).reshape(npix, 1)

    dwdp_x = np.empty((npix, 8), dtype=T.dtype)
    dwdp_y = np.empty((npix, 8), dtype=T.dtype)
    row = 0
    for y in range(theight):
        for x in range(twidth):
            dwdp_x[row] = np.array([ x, y, 1, 0, 0, 0, -x*x, -x*y ])
            dwdp_y[row] = np.array([ 0, 0, 0, x, y, 1, -x*y, -y*y ])
            row += 1

    ## row-wise multiply and element-wise add
    J = Tx * dwdp_x + Ty * dwdp_y

    JtJ = np.dot(J.T, J)

    return J, JtJ


def track_homography_lk(image, homography_p, T, J, JtJ, max_iter=50):
    theight, twidth = T.shape
    npix = twidth * theight

    for iter in range(max_iter):
        Ip = cv2.warpPerspective(image, inv(homography_p), (twidth, theight))
        Ip = np.float32(Ip)
        err = (Ip - T).reshape(npix)
        dp = np.linalg.solve(JtJ, np.dot(J.T, err))
        homography_dp = np.array([[1 + dp[0], dp[1], dp[2]],
                                 [dp[3], 1 + dp[4], dp[5]],
                                 [dp[6], dp[7], 1.0]])
        homography_p = np.dot(homography_p, inv(homography_dp))

    return homography_p


def main():
    cap = ic.select_capture_source(sys.argv)

    mstate = {
        'selection': 'invalid',
        'xybegin': (-1, -1),
        'xyend': (-1, -1),
    }
    cv2.namedWindow('track')
    cv2.setMouseCallback('track', ic.on_mouse_rect, mstate)

    target = None

    while True:
        grabbed, frame_color = cap.read()
        if not grabbed:
            break
        frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

        if mstate['selection'] == 'valid' and target is None:
            ## initialization requested

            xybegin = np.array(mstate['xybegin'])
            xyend = np.array(mstate['xyend'])
            xycenter = np.int16((xybegin + xyend) / 2)
            tsize = xyend - xybegin   ## == (width, height)
            target = cv2.getRectSubPix(frame, tsize, xycenter)
            T = np.float32(target)
            J, JtJ = compute_derivatives(T)

            tpnts = np.float32([[0, 0],
                                [0, tsize[1]],
                                [tsize[0], tsize[1]],
                                [tsize[0], 0]
                                ])
            ipnts = np.float32(tpnts + xybegin)
            homography_p = cv2.getPerspectiveTransform(tpnts, ipnts)

            cv2.imshow('template', target)


        if mstate['selection'] == 'valid':
            assert target is not None
            assert 'homography_p' in locals()
            assert 'T' in locals()
            assert 'J' in locals()
            assert 'JtJ' in locals()
            assert 'tpnts' in locals()

            try:
                homography_p = track_homography_lk(frame, homography_p,
                                                   T, J, JtJ)
                ipnts = cv2.perspectiveTransform(np.float32([tpnts]),
                                                 homography_p)
                cv2.polylines(frame_color, np.int32(ipnts), 1, (0, 0, 255), 2)
            except np.linalg.linalg.LinAlgError:
                homography_p = None

        elif mstate['selection'] == 'ongoing':
            target = None

            xybegin = mstate['xybegin']
            xyend = mstate['xyend']
            cv2.rectangle(frame_color, xybegin, xyend, (255, 0, 0), 2)

        elif mstate['selection'] == 'invalid':
            target = None

        cv2.imshow('track', frame_color)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
