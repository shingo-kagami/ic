import sys
import cv2
import numpy as np

import ic_utils as ic


def main():
    cap = ic.select_capture_source(sys.argv)
    grabbed, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        grabbed, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not grabbed:
            break

        diff_img = np.int8((np.int16(frame) - np.int16(prev_frame)) / 2)
        prev_frame = frame

        ## cv2.imshow adds 128 to pixel values when it receives np.int8 image
        cv2.imshow('result', diff_img)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
