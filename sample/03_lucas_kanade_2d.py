import sys
import cv2
import numpy as np
from numba import jit

def onMouse(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0] = (x, y)
        param[1] = True

def precomputeDerivatives(T):
    theight = T.shape[0]
    twidth = T.shape[1]

    Tx = np.zeros((theight, twidth), dtype=np.float32)
    Ty = np.zeros((theight, twidth), dtype=np.float32)
    TxTx = np.zeros((theight, twidth), dtype=np.float32)
    TyTy = np.zeros((theight, twidth), dtype=np.float32)
    TxTy = np.zeros((theight, twidth), dtype=np.float32)
    Hessian = np.zeros((2, 2), dtype=np.float32)

    for j in range(1, theight - 1):
        for i in range(1, twidth - 1):
            Tx[j, i] = (T[j, i + 1] - T[j, i - 1]) / 2
            Ty[j, i] = (T[j + 1, i] - T[j - 1, i]) / 2
            TxTx[j, i] = Tx[j, i] * Tx[j, i]
            TyTy[j, i] = Ty[j, i] * Ty[j, i]
            TxTy[j, i] = Tx[j, i] * Ty[j, i]
            Hessian[0, 0] += TxTx[j, i]
            Hessian[1, 1] += TyTy[j, i]
            Hessian[0, 1] += TxTy[j, i]
    Hessian[1, 0] = Hessian[0, 1]

    return Tx, Ty, Hessian


@jit
def computeJe(Iw, T, Tx, Ty):
    theight = Iw.shape[0]
    twidth = Iw.shape[1]

    e = np.zeros((theight, twidth), dtype=np.float32)
    Tx_e = np.zeros((theight, twidth), dtype=np.float32)
    Ty_e = np.zeros((theight, twidth), dtype=np.float32)
    Je = np.zeros((2,))

    for j in range(1, theight - 1):
        for i in range(1, twidth - 1):
            e[j, i] = Iw[j, i] - T[j, i]
            Tx_e[j, i] = Tx[j, i] * e[j, i]
            Ty_e[j, i] = Ty[j, i] * e[j, i]
            Je[0] += Tx_e[j, i]
            Je[1] += Ty_e[j, i]

    return Je


def matchTemplateLK(image, tcenter, T, Tx, Ty, Hessian):
    max_iter = 20
    theight = T.shape[0]
    twidth = T.shape[1]

    for iter in range(max_iter):
        Iw = cv2.getRectSubPix(image, (twidth, theight), tcenter)
        Iw = np.float32(Iw)
        Je = computeJe(Iw, T, Tx, Ty)
        p = np.linalg.solve(Hessian, Je)
        tcenter = (tcenter[0] - p[0], tcenter[1] - p[1])
        if np.linalg.norm(p) < 0.2:
            break

    return tcenter


if __name__ == '__main__':
    cv2.namedWindow('track')

    twidth = 64
    theight = 64

    cap_src = 'vtest.avi'
    if len(sys.argv) == 2:
        if sys.argv[1].isdecimal():
            cap_src = int(sys.argv[1])
        else:
            cap_src = sys.argv[1]
    cap = cv2.VideoCapture(cap_src)

    retval, input_prev = cap.read()
    input_prev = cv2.cvtColor(input_prev, cv2.COLOR_BGR2GRAY)

    print("click to set the template")

    mouse_param = [ (-1, -1), False ]

    while True:
        retval, input_color = cap.read()
        input = cv2.cvtColor(input_color, cv2.COLOR_BGR2GRAY)

        cv2.setMouseCallback('track', onMouse, mouse_param)

        tcenter = mouse_param[0]
        is_clicked = mouse_param[1]

        if is_clicked:
            tmplt = cv2.getRectSubPix(input_prev, (twidth, theight), tcenter)
            T = np.float32(tmplt)
            Tx, Ty, Hessian = precomputeDerivatives(T)
            mouse_param[1] = False

        if tcenter != (-1, -1):
            x0 = int(tcenter[0])
            y0 = int(tcenter[1])

            tcenter = matchTemplateLK(input, tcenter, T, Tx, Ty, Hessian)

            cv2.rectangle(input_color, 
                          (int(tcenter[0] - twidth / 2),
                           int(tcenter[1] - theight / 2)),
                          (int(tcenter[0] + twidth / 2),
                           int(tcenter[1] + theight / 2)), 
                          (0, 0, 255), 3)
            cv2.imshow("template", tmplt)
            
        input_prev = input.copy()
        mouse_param[0] = tcenter

        cv2.imshow("track", input_color)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
