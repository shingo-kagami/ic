import sys
import cv2
import numpy as np
from numba import jit

import ic_utils as ic


@jit
def SSD(target, candidate):
    height, width = target.shape
    ssd_val = 0

    for j in range(height):
        for i in range(width):
            d = candidate[j, i] - target[j, i]
            ssd_val += d * d

    return ssd_val


@jit
def match_template(image, target, current_center, search_margin):
    iheight, iwidth = image.shape
    theight, twidth = target.shape

    ## top-left corner of current candidate region
    cxbegin = current_center[0] - int(twidth / 2)
    cybegin = current_center[1] - int(theight / 2)

    ## top-left and bottom-right corners of search area
    sxbegin = max(0, cxbegin - search_margin)
    sybegin = max(0, cybegin - search_margin)
    sxend = min(cxbegin + search_margin, iwidth - twidth)
    syend = min(cybegin + search_margin, iheight - theight)

    min_ssd = sys.maxsize  ## initialized with a large, large number
    for j in range(sybegin, syend):
        for i in range(sxbegin, sxend):
            candidate = image[j:(j + theight), i:(i + twidth)]
            ssd = SSD(target, candidate)
            if ssd < min_ssd:
                min_ssd = ssd
                min_location = (i, j)

    new_current_center = (min_location[0] + int(twidth / 2), 
                          min_location[1] + int(theight / 2))

    return np.int16(new_current_center)


def main():
    search_margin = 32
    cap = ic.select_capture_source(sys.argv)

    mstate = {
        "selection": "invalid",
        "xybegin": (-1, -1),
        "xyend": (-1, -1),
    }
    cv2.namedWindow('track')
    cv2.setMouseCallback('track', ic.on_mouse_rect, mstate)

    cv2.createTrackbar('my; cv', 'track', 0, 1, ic.do_nothing)

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

            cv2.imshow("template", target)


        if mstate['selection'] == 'valid':
            ## make sure target, current_center and tsize have been initialized
            assert target is not None
            assert 'current_center' in locals()
            assert 'tsize' in locals()

            impl = cv2.getTrackbarPos('my; cv', 'track')
            if impl == 0:
                current_center = match_template(frame, target, current_center,
                                                search_margin)
                cxybegin = np.int16(current_center - tsize/2)
                cxyend = np.int16(current_center + tsize/2)
            else:
                search_area = cv2.getRectSubPix(frame, tsize + 2*search_margin,
                                                current_center)
                dissim = cv2.matchTemplate(search_area, target, cv2.TM_SQDIFF)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dissim)
                sxybegin = current_center - tsize/2 - search_margin
                cxybegin = np.int16(sxybegin + min_loc)
                cxyend = cxybegin + tsize
                current_center = (cxybegin + cxyend) / 2

            cv2.rectangle(frame_color, cxybegin, cxyend,
                          color=(0, 0, 255), thickness=3)
            cv2.rectangle(frame_color,
                          cxybegin - search_margin,
                          cxyend + search_margin,
                          color=(0, 255, 0), thickness=1)

        elif mstate['selection'] == 'ongoing':
            target = None

            xybegin = mstate['xybegin']
            xyend = mstate['xyend']
            cv2.rectangle(frame_color, xybegin, xyend,
                          color=(255, 0, 0), thickness=2)

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
