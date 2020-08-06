import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from numba import jit

# Make sure you have "color_histogram_utils.py" in the same directory
from color_histogram_utils import createHueSatHistogram, createHueSatLegend, plotHueSatHistogram

@jit
def createSimilarityMap(input_hsv, hist_model, tsize, step, is_weighted):
    iwidth = input_hsv.shape[1]
    iheight = input_hsv.shape[0]
    twidth = tsize[0]
    theight = tsize[1]
    hist_size = (hist_model.shape[1], hist_model.shape[0])

    similarity_map = np.zeros(((iheight - theight + 1),
                               (iwidth - twidth + 1)), 
                              np.float64)
    swidth = similarity_map.shape[1]
    sheight = similarity_map.shape[0]

    for j in range(0, sheight, step):
        for i in range(0, swidth, step):
            region = input_hsv[j:(j + theight), 
                               i:(i + twidth)]
            hist = createHueSatHistogram(region, hist_size, is_weighted)
            bhattacharyya_coeff = np.sum(np.sqrt(hist_model * hist))
            similarity_map[j:(j + step), i:(i + step)] = bhattacharyya_coeff

    return similarity_map


def plotSimilarityMap(ax, similarity_map, step):
    cols = np.arange(0, similarity_map.shape[1], step)
    rows = np.arange(0, similarity_map.shape[0], step)
    mesh_rows, mesh_cols = np.meshgrid(rows, cols)

    # x and y are exchanged so that the direction of y agrees with an image
    z = similarity_map[::step, ::step].T
    ax.plot_surface(mesh_rows, mesh_cols, z, cmap=cm.coolwarm)
    ax.set_ylabel('X')
    ax.set_xlabel('Y')


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
    cv2.namedWindow('input')

    cap_src = 'vtest.avi'
    if len(sys.argv) == 2:
        if sys.argv[1].isdecimal():
            cap_src = int(sys.argv[1])
        else:
            cap_src = sys.argv[1]
    cap = cv2.VideoCapture(cap_src)

    twidth = 64
    theight = 64
    tsize = (twidth, theight)

    hbin = 64
    sbin = 64
    hist_size = (hbin, sbin)

    hist_weighted = True
    #hist_weighted = False

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax_hist = fig.add_subplot(1, 2, 1, projection='3d')
    ax_hist.view_init(50, -30)
    ax_sim = fig.add_subplot(1, 2, 2, projection='3d')
    ax_sim.view_init(50, -30)

    hs_legend = createHueSatLegend(hbin, sbin)

    state = { "tracking": "off", 
              "topleft": (-1, -1), 
              "bottomright": (-1, -1) }
    cv2.namedWindow("input")
    cv2.setMouseCallback("input", onMouse, state)

    print("drag to set the model region")

    while True:
        retval, input_bgr = cap.read()
        input_hsv = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2HSV)

        if state["tracking"] == "init_requested":
            tcenter = getCenter(state["topleft"], state["bottomright"])
            twidth = getWidth(state["topleft"], state["bottomright"])
            theight = getHeight(state["topleft"], state["bottomright"])
            target_bgr = cv2.getRectSubPix(input_bgr, (twidth, theight), tcenter)
            cv2.imshow("target", target_bgr)
            target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)
            hist_model = createHueSatHistogram(target_hsv, hist_size, hist_weighted)
            ax_hist.clear()
            plotHueSatHistogram(ax_hist, hist_model, hs_legend)
            state["tracking"] = "on"

        elif state["tracking"] == "on":
            step = 8
            similarity_map = createSimilarityMap(input_hsv, hist_model, tsize, 
                                                 step, hist_weighted)
            ax_sim.clear()
            plotSimilarityMap(ax_sim, similarity_map, step)
            plt.pause(0.001)

        elif state["tracking"] == "selecting":
            center = getCenter(state["topleft"], state["bottomright"])
            w = getWidth(state["topleft"], state["bottomright"])
            h = getHeight(state["topleft"], state["bottomright"])
            if w > 0 and h > 0:
              cv2.ellipse(input_bgr, (int(center[0]), int(center[1])),
                          (int(w / 2), int(h / 2)), 
                          0, 0, 360, (0, 255, 0), 1)

        cv2.imshow("input", input_bgr)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
