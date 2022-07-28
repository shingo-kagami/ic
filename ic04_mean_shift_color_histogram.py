import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from numba import jit


import ic_utils as ic
from ic04_color_histogram import generate_hue_sat_histogram


@jit
def mean_shift_vector(image_area, hist_model, hist_cand):
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
            at_sbin = int(sat * sbin / 256.0)
            at_hbin = int(hue * hbin / 180.0)
            q = hist_model[at_sbin, at_hbin]
            p = hist_cand[at_sbin, at_hbin]
            x_norm = (i - cx) * w_normalizer
            y_norm = (j - cy) * h_normalizer
            rr = x_norm ** 2 + y_norm ** 2
            if rr < 1 and p != 0.0:
                w = math.sqrt(q / p)
                m0 += w
                mx += w * i
                my += w * j

    if m0 == 0:
        return 0.0, 0.0
    else:
        return mx/m0 - cx, my/m0 - cy


def track_by_mean_shift(image, tcenter, tsize, hist_model, max_iter=20):
    twidth = tsize[0]
    theight = tsize[1]
    hist_size = (hist_model.shape[1], hist_model.shape[0])

    for iter in range(max_iter):
        candidate = cv2.getRectSubPix(image, (twidth, theight), tcenter)
        hist_cand = generate_hue_sat_histogram(candidate, hist_size, True)
        msv_x, msv_y = mean_shift_vector(candidate, hist_model, hist_cand)
        tcenter = (tcenter[0] + msv_x, tcenter[1] + msv_y)
        dd = msv_x ** 2 + msv_y ** 2
        if dd < 1.0:
            break

    bhattacharyya_coeff = np.sum(np.sqrt(hist_model * hist_cand))
    return np.int16(tcenter), bhattacharyya_coeff


def main():
    hbin = 64
    sbin = 64
    hist_size = (hbin, sbin)

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
            hist_model = generate_hue_sat_histogram(target_hsv, hist_size, True)

            cv2.imshow('model', target)


        if mstate['selection'] == 'valid':
            current_center, bc = track_by_mean_shift(frame_hsv,
                                                     current_center, tsize,
                                                     hist_model)
            cv2.ellipse(frame_bgr, current_center, np.int16(tsize/2),
                        0, 0, 360, (0, 0, int(255 * bc)), 3)

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


        cv2.imshow('track', frame_bgr)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
