import sys
import cv2
import numpy as np
from numba import jit

import ic_utils as ic


@jit
def compute_derivatives(T):
    Tx = np.zeros_like(T)
    Ty = np.zeros_like(T)
    TxTx = np.zeros_like(T)
    TyTy = np.zeros_like(T)
    TxTy = np.zeros_like(T)
    H = np.zeros((2, 2), dtype=T.dtype)

    theight, twidth = T.shape

    for j in range(1, theight - 1):
        for i in range(1, twidth - 1):
            Tx[j, i] = (T[j, i + 1] - T[j, i - 1]) / 2
            Ty[j, i] = (T[j + 1, i] - T[j - 1, i]) / 2
            TxTx[j, i] = Tx[j, i] * Tx[j, i]
            TyTy[j, i] = Ty[j, i] * Ty[j, i]
            TxTy[j, i] = Tx[j, i] * Ty[j, i]
            H[0, 0] += TxTx[j, i]
            H[1, 1] += TyTy[j, i]
            H[0, 1] += TxTy[j, i]
    H[1, 0] = H[0, 1]

    return Tx, Ty, H


@jit
def compute_Jt_err(Ip, T, Tx, Ty):
    err = np.zeros_like(T)
    Tx_err = np.zeros_like(T)
    Ty_err = np.zeros_like(T)
    Jt_err = np.zeros((2,), dtype=T.dtype)

    theight, twidth = T.shape

    for j in range(1, theight - 1):
        for i in range(1, twidth - 1):
            err[j, i] = Ip[j, i] - T[j, i]
            Tx_err[j, i] = Tx[j, i] * err[j, i]
            Ty_err[j, i] = Ty[j, i] * err[j, i]
            Jt_err[0] += Tx_err[j, i]
            Jt_err[1] += Ty_err[j, i]

    return Jt_err


def match_template_lk(image, current_center, T, Tx, Ty, JtJ, max_iter=50):
    theight, twidth = T.shape

    for iter in range(max_iter):
        Ip = cv2.getRectSubPix(image, (twidth, theight), current_center)
        Ip = np.float32(Ip)
        Jt_err = compute_Jt_err(Ip, T, Tx, Ty)
        dp = np.linalg.solve(JtJ, Jt_err)
        current_center = (current_center[0] - dp[0], current_center[1] - dp[1])
        if np.linalg.norm(dp) < 0.2:
            break

    return current_center


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
            current_center = np.int16((xybegin + xyend) / 2)
            tsize = xyend - xybegin   ## == (width, height)
            target = cv2.getRectSubPix(frame, tsize, current_center)
            T = np.float32(target)
            Tx, Ty, JtJ = compute_derivatives(T)

            cv2.imshow('template', target)


        if mstate['selection'] == 'valid':
            assert target is not None
            assert 'current_center' in locals()
            assert 'T' in locals()
            assert 'Tx' in locals()
            assert 'Ty' in locals()
            assert 'JtJ' in locals()
            assert 'tsize' in locals()

            current_center = match_template_lk(frame, current_center,
                                               T, Tx, Ty, JtJ)
            cxybegin = np.int16(current_center - tsize / 2)
            cxyend = np.int16(current_center + tsize / 2)

            cv2.rectangle(frame_color, cxybegin, cxyend, (0, 0, 255), 3)

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
