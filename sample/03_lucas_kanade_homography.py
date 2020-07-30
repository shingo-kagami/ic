import sys
import cv2
import numpy as np
from numpy.linalg import inv

def onMouse(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0] = (x, y)
        param[1] = True

def trackHomographyLK(image, tmplt, G):
    max_iter = 20

    theight = tmplt.shape[0]
    twidth = tmplt.shape[1]
    npix = twidth * theight

    T = np.float32(tmplt)
    grad_x = np.gradient(T, axis=1).reshape(npix, 1)
    grad_y = np.gradient(T, axis=0).reshape(npix, 1)

    dwdp_x = np.empty((npix, 8))
    dwdp_y = np.empty((npix, 8))
    row = 0
    for y in range(theight):
        for x in range(twidth):
            dwdp_x[row] = np.array([ x, y, 1, 0, 0, 0, -x*x, -x*y ])
            dwdp_y[row] = np.array([ 0, 0, 0, x, y, 1, -x*y, -y*y ])
            row += 1

    # row-wise multiply and element-wise add
    J = grad_x * dwdp_x + grad_y * dwdp_y

    Hessian = np.dot(J.T, J)

    for iter in range(max_iter):
        Iw = cv2.warpPerspective(image, inv(G), (T.shape[1], T.shape[0]))
        Iw = np.float32(Iw)
        e = (Iw - T).reshape(npix)
        
        p = np.linalg.solve(Hessian, np.dot(J.T, e))
        G_p = np.array([[1 + p[0], p[1], p[2]],
                        [p[3], 1 + p[4], p[5]],
                        [p[6], p[7], 1.0]])
        G = np.dot(G, inv(G_p))

    return G


if __name__ == '__main__':
    cv2.namedWindow('track')

    twidth = 96
    theight = 96
    tpnts = np.float32([[0, 0],
                        [0, theight -1],
                        [twidth - 1, theight - 1],
                        [twidth - 1, 0]
    ])

    cap_src = 'vtest.avi'
    if len(sys.argv) == 2:
        if sys.argv[1].isdecimal():
            cap_src = int(sys.argv[1])
        else:
            cap_src = sys.argv[1]
    cap = cv2.VideoCapture(cap_src)

    print("click to set the template")

    mouse_param = [ (-1, -1), False ]

    while True:
        retval, input_color = cap.read()
        input = cv2.cvtColor(input_color, cv2.COLOR_BGR2GRAY)

        cv2.setMouseCallback('track', onMouse, mouse_param)

        tcenter = mouse_param[0]
        is_clicked = mouse_param[1]
        if is_clicked:
            tmplt = cv2.getRectSubPix(input, (twidth, theight), tcenter)
            ipnts = np.float32([
                [tcenter[0] - twidth / 2, tcenter[1] - theight / 2],
                [tcenter[0] - twidth / 2, tcenter[1] + theight / 2],
                [tcenter[0] + twidth / 2, tcenter[1] + theight / 2],
                [tcenter[0] + twidth / 2, tcenter[1] - theight / 2]
            ])
            G = cv2.getPerspectiveTransform(tpnts, ipnts)
            mouse_param[1] = False

        if tcenter != (-1, -1):
            try:
                G = trackHomographyLK(input, tmplt, G)
                ipnts = cv2.perspectiveTransform(np.float32([tpnts]), G)
                cv2.polylines(input_color, np.int32(ipnts), 1, (0, 0, 255), 2)
                cv2.imshow("template", tmplt)
            except np.linalg.linalg.LinAlgError as e:
                mouse_param = [ (-1, -1), False ]

        cv2.imshow("track", input_color)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
