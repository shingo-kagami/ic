import sys
import cv2
import numpy as np

import ic_utils as ic


def main():
    cap = ic.select_capture_source(sys.argv)

    grabbed, img = cap.read()
    img_size = (img.shape[1], img.shape[0])

    ## model file from: https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
    face_detector = cv2.FaceDetectorYN_create('face_detection_yunet_2022mar.onnx', '', img_size)

    while True:
        grabbed, img = cap.read()
        if not grabbed:
            break

        retval, faces = face_detector.detect(img)

        if faces is None:
            faces = []

        for face in faces:
            coordinates = np.int32(face[0:14])
            confidence = face[14]

            x0, y0, w, h = coordinates[0:4]
            reye = coordinates[4:6]
            leye = coordinates[6:8]
            nose = coordinates[8:10]
            rmouse = coordinates[10:12]
            lmouse = coordinates[12:14]

            cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 2)
            for pos in (reye, leye, nose, rmouse, lmouse):
                cv2.circle(img, pos, 2, (0, 255, 0), -1)

        cv2.imshow('result', img)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
