import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from numba import jit

import ic_utils as ic
from ic04_color_histogram import (generate_hue_sat_histogram,
                                  generate_hue_sat_legend,
                                  plot_hue_sat_histogram)

@jit
def similarity_map(input_hsv, hist_model, tsize, step, is_weighted):
    iwidth = input_hsv.shape[1]
    iheight = input_hsv.shape[0]
    twidth = tsize[0]
    theight = tsize[1]
    hist_size = (hist_model.shape[1], hist_model.shape[0])

    sim_map = np.zeros(((iheight - theight + 1),
                        (iwidth - twidth + 1)), dtype=np.float64)
    swidth = sim_map.shape[1]
    sheight = sim_map.shape[0]

    for j in range(0, sheight, step):
        for i in range(0, swidth, step):
            region = input_hsv[j:(j + theight), i:(i + twidth)]
            hist = generate_hue_sat_histogram(region, hist_size, is_weighted)
            bhattacharyya_coeff = np.sum(np.sqrt(hist_model * hist))
            sim_map[j:(j + step), i:(i + step)] = bhattacharyya_coeff

    return sim_map


def plot_similarity_map(ax, sim_map, step):
    cols = np.arange(0, sim_map.shape[1], step)
    rows = np.arange(0, sim_map.shape[0], step)
    mesh_rows, mesh_cols = np.meshgrid(rows, cols)

    # x and y are exchanged so that the direction of y agrees with an image
    z = sim_map[::step, ::step].T
    ax.plot_surface(mesh_rows, mesh_cols, z, cmap=cm.coolwarm)
    ax.set_ylabel('X')
    ax.set_xlabel('Y')


def main():
    hbin = 64
    sbin = 64
    hist_size = (hbin, sbin)
    hist_weighted = True

    cap = ic.select_capture_source(sys.argv)

    mstate = {
        'selection': 'invalid',
        'xybegin': (-1, -1),
        'xyend': (-1, -1),
    }

    cv2.namedWindow('input')
    cv2.setMouseCallback('input', ic.on_mouse_rect, mstate)

    target = None

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax_hist = fig.add_subplot(1, 2, 1, projection='3d')
    ax_hist.view_init(50, -30)
    ax_sim = fig.add_subplot(1, 2, 2, projection='3d')
    ax_sim.view_init(50, -30)
    hs_legend = generate_hue_sat_legend(hbin, sbin)

    while True:
        grabbed, frame_bgr = cap.read()
        if not grabbed:
            break
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        if mstate['selection'] == 'valid' and target is None:
            ## initialization requested

            xybegin = np.array(mstate['xybegin'])
            xyend = np.array(mstate['xyend'])
            current_center = np.int16((xybegin + xyend) / 2)

            tsize = xyend - xybegin   ## == (width, height)
            target = cv2.getRectSubPix(frame_bgr, tsize, current_center)
            target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
            hist_model = generate_hue_sat_histogram(target_hsv, hist_size,
                                                    hist_weighted)
            cv2.imshow('target', target)
            ax_hist.clear()
            plot_hue_sat_histogram(ax_hist, hist_model, hs_legend)


        if mstate['selection'] == 'valid':
            step = 8
            sim_map = similarity_map(frame_hsv, hist_model, tsize, 
                                     step, hist_weighted)
            ax_sim.clear()
            plot_similarity_map(ax_sim, sim_map, step)
            plt.pause(0.001)

        elif mstate['selection'] == 'ongoing':
            target = None

            xybegin = np.array(mstate['xybegin'])
            xyend = np.array(mstate['xyend'])
            select_center = np.int16((xybegin + xyend) / 2)
            select_size = xyend - xybegin

            if select_size[0] > 0 and select_size[1] > 0:
                cv2.ellipse(frame_bgr, select_center, np.int16(select_size/2),
                            0, 0, 360, (0, 255, 0), 1)

        elif mstate['selection'] == 'invalid':
            target = None


        cv2.imshow('input', frame_bgr)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
