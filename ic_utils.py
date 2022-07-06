import cv2
import numpy as np


def select_capture_source(argv):
    cap_src = 'vtest.avi'
    if len(argv) == 2:
        if argv[1].isdecimal():
            cap_src = int(argv[1])
        else:
            cap_src = argv[1]
    return cv2.VideoCapture(cap_src)


def do_nothing(x):
    pass


def on_mouse_rect(event, x, y, flag, mstate):
    if event == cv2.EVENT_LBUTTONDOWN:
        mstate['selection'] = 'ongoing'
        mstate['xybegin'] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        width = mstate['xyend'][0] - mstate['xybegin'][0]
        height = mstate['xyend'][1] - mstate['xybegin'][1]
        if width > 0 and height > 0:
            mstate['selection'] = 'valid'
        else:
            mstate['selection'] = 'invalid'
    elif event == cv2.EVENT_RBUTTONDOWN:
        mstate['selection'] = 'invalid'

    if mstate['selection'] == 'ongoing':
        mstate['xyend'] = (x, y)


def putText(img, text, pos, color, scale=1.0):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_PLAIN, scale, color)
