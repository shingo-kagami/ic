import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from numba import jit

# Make sure you have "color_histogram_utils.py" in the same directory
from color_histogram_utils import createHueSatHistogram, createHueSatLegend, plotHueSatHistogram

@jit
def meanShiftVector(image_area, hist_model, hist_cand):
    width =  image_area.shape[1]
    height =  image_area.shape[0]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    hbin = hist_model.shape[1]
    sbin = hist_model.shape[0]
    w_normalizer = 2.0 / width
    h_normalizer = 2.0 / height

    m0 = mx = my = 0.0
    for j in range(height):
        for i in range(width):
            hue = image_area[j, i, 0]
            sat = image_area[j, i, 1]
            q = hist_model[int(sat * sbin / 256.0), int(hue * hbin / 180.0)]
            p = hist_cand[int(sat * sbin / 256.0), int(hue * hbin / 180.0)]
            x_norm = (i - cx) * w_normalizer
            y_norm = (j - cy) * h_normalizer
            rr = x_norm * x_norm + y_norm * y_norm
            if rr < 1 and p != 0.0:
                w = math.sqrt(q / p)
                m0 += w
                mx += w * i
                my += w * j
    if m0 == 0:
        return 0.0, 0.0
    else:
        return mx/m0 - cx, my/m0 - cy
    

def trackByMeanShift(image, tcenter, tsize, hist_model, hist_size):
    iwidth = image.shape[1]
    iheight = image.shape[0]
    twidth = tsize[0]
    theight = tsize[1]
    w_normalizer = 2.0 / twidth
    h_normalizer = 2.0 / theight
    half_w = int(twidth / 2)
    half_h = int(theight / 2)
    hbin = hist_size[0]
    sbin = hist_size[1]
    max_iter = 20

    for iter in range(max_iter):
        candidate = cv2.getRectSubPix(image, (twidth, theight), tcenter)
        hist_cand = createHueSatHistogram(candidate, hist_size, True)
        mv_x, mv_y = meanShiftVector(candidate, hist_model, hist_cand)
        tcenter = (tcenter[0] + mv_x, tcenter[1] + mv_y)
        dd = mv_x * mv_x + mv_y * mv_y 
        if dd < 1.0:
            break

    bhattacharyya_coeff = np.sum(np.sqrt(hist_model * hist_cand))
        
    return tcenter, hist_cand, bhattacharyya_coeff


def getCenter(topleft, bottomright):
    return ((topleft[0] + bottomright[0]) / 2, (topleft[1] + bottomright[1]) / 2)

def getWidth(topleft, bottomright):
    return bottomright[0] - topleft[0]

def getHeight(topleft, bottomright):
    return bottomright[1] - topleft[1]

def onMouse(event, x, y, flag, state):
    if state["tracking"] == "selecting":
        state["bottomright"] = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        state["tracking"] = "selecting"
        state["topleft"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if (getWidth(state["topleft"], state["bottomright"]) > 0 and
            getHeight(state["topleft"], state["bottomright"]) > 0):
            state["tracking"] = "init_requested"
        else:
            state["tracking"] = "off"


if __name__ == '__main__':
    #plot_enabled = True
    plot_enabled = False

    cv2.namedWindow('track')

    cap_src = 'vtest.avi'
    if len(sys.argv) == 2:
        if sys.argv[1].isdecimal():
            cap_src = int(sys.argv[1])
        else:
            cap_src = sys.argv[1]
    cap = cv2.VideoCapture(cap_src)

    hbin = 64
    sbin = 64
    hist_size = (hbin, sbin)

    if plot_enabled:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax_model = fig.add_subplot(1, 2, 1, projection='3d')
        ax_model.view_init(50, -30)
        ax_cand = fig.add_subplot(1, 2, 2, projection='3d')
        ax_cand.view_init(50, -30)
        hs_legend = createHueSatLegend(hbin, sbin)

    state = { "tracking": "off", 
              "topleft": (-1, -1), 
              "bottomright": (-1, -1) }
    cv2.namedWindow("track")
    cv2.setMouseCallback("track", onMouse, state)

    print("drag to set the model region")

    while True:
        retval, input_bgr = cap.read()
        input_disp = input_bgr.copy()
        input_hsv = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2HSV)

        if state["tracking"] == "init_requested":
            tcenter = getCenter(state["topleft"], state["bottomright"])
            twidth = getWidth(state["topleft"], state["bottomright"])
            theight = getHeight(state["topleft"], state["bottomright"])
            tsize = (twidth, theight)
            
            target_bgr = cv2.getRectSubPix(input_bgr, tsize, tcenter)
            target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)
            hist_model = createHueSatHistogram(target_hsv, hist_size, True)
            cv2.imshow("model", target_bgr)
            if plot_enabled:
                ax_model.clear()
                ax_model.set_title("model histogram")
                plotHueSatHistogram(ax_model, hist_model, hs_legend)
            state["tracking"] = "on"

        elif state["tracking"] == "on":
            tcenter, hist_cand, bc = trackByMeanShift(input_hsv, tcenter, tsize, 
                                                      hist_model, hist_size)
            cv2.ellipse(input_disp, (int(tcenter[0]), int(tcenter[1])),
                        (int(twidth / 2), int(theight / 2)), 
                        0, 0, 360, (0, 0, int(255 * bc)), 3)
            if plot_enabled:
                ax_cand.clear()
                ax_cand.set_title("candidate histogram")
                plotHueSatHistogram(ax_cand, hist_cand, hs_legend)
                plt.pause(0.001)

        elif state["tracking"] == "selecting":
            center = getCenter(state["topleft"], state["bottomright"])
            w = getWidth(state["topleft"], state["bottomright"])
            h = getHeight(state["topleft"], state["bottomright"])
            if w > 0 and h > 0:
              cv2.ellipse(input_disp, (int(center[0]), int(center[1])),
                          (int(w / 2), int(h / 2)), 
                          0, 0, 360, (0, 255, 0), 1)

        cv2.imshow("track", input_disp)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
