import sys
import cv2
import numpy as np
from numba import jit

def onMouse(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0] = (x, y)
        param[1] = True

@jit
def SSD(tmplt, candidate):
    ssd_val = 0
    height = tmplt.shape[0]
    width = tmplt.shape[1]
    for j in range(height):
        for i in range(width):
            d = int(candidate[j, i]) - int(tmplt[j, i])
            ssd_val += d * d
    return ssd_val

@jit
def matchTemplate(image, tmplt, tcenter, search_size):
    iheight = image.shape[0]
    iwidth = image.shape[1]
    theight = tmplt.shape[0]
    twidth = tmplt.shape[1]

    sxmin = max(0, tcenter[0] - int(search_size[0] / 2))
    sxmax = min(iwidth - twidth - 1, tcenter[0] + int(search_size[0] / 2))
    symin = max(0, tcenter[1] - int(search_size[1] / 2))
    symax = min(iheight - theight - 1, tcenter[1] + int(search_size[1] / 2))

    min_ssd = sys.maxsize
    for j in range(symin, symax + 1):
        for i in range(sxmin, sxmax + 1):
            candidate = image[j:(j + theight), i:(i + twidth)]
            ssd = SSD(tmplt, candidate)
            if ssd < min_ssd:
                min_ssd = ssd
                min_location = (i, j)

    new_tcenter = (min_location[0] + int(twidth / 2), 
                   min_location[1] + int(theight / 2))

    return new_tcenter


if __name__ == '__main__':
    cv2.namedWindow('track')

    twidth = 64
    theight = 64
    search_margin = 32

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

        if is_clicked:  # template initialization requested
            tmplt = cv2.getRectSubPix(input_prev, (twidth, theight), tcenter)
            mouse_param[1] = False  # request completed

        if tcenter != (-1, -1):
            tcenter = matchTemplate(input, tmplt, tcenter, 
                                    (twidth + 2 * search_margin,
                                     theight + 2 * search_margin))
            cv2.rectangle(input_color, 
                          (tcenter[0] - int(twidth / 2),
                           tcenter[1] - int(theight / 2)),
                          (tcenter[0] + int(twidth / 2),
                           tcenter[1] + int(theight / 2)), 
                          (0, 0, 255), 3)
            cv2.rectangle(input_color, 
                          (tcenter[0] - int(twidth / 2) - search_margin,
                           tcenter[1] - int(theight / 2) - search_margin),
                          (tcenter[0] + int(twidth / 2) + search_margin,
                           tcenter[1] + int(theight / 2) + search_margin), 
                          (0, 255, 0), 1)
            cv2.imshow("template", tmplt)
            
        input_prev = input.copy()
        mouse_param[0] = tcenter

        cv2.imshow("track", input_color)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
