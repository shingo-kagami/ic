import cv2
import sys
import numpy as np

def getGrayFrame(cap):
    retval, input = cap.read()
    if retval == False:
        return False, np.array()
    else:
        return True, cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

if __name__ == '__main__':
    cap_src = 'vtest.avi';
    if len(sys.argv) == 2:
        if sys.argv[1].isdecimal():
            cap_src = int(sys.argv[1])
        else:
            cap_src = sys.argv[1]
    cap = cv2.VideoCapture(cap_src)

    retval, prev_frame = getGrayFrame(cap)

    while True:
        retval, input = getGrayFrame(cap)
        if retval == False:
            break
        diff_img = np.uint8((np.int16(input) - 
                             np.int16(prev_frame)) / 2 + 128)
        prev_frame = input

        cv2.imshow("test", diff_img)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
