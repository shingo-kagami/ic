import sys

import cv2
import numpy as np
from ultralytics import YOLO

import ic_utils as ic

def main():
    model = YOLO("yolov8n.pt")
    cap = ic.select_capture_source(sys.argv)

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        img_disp = frame.copy()
        #results = model(source=frame, verbose=True, show=True)
        results = model(source=frame, verbose=False, show=False)
        #for r in results: # loop through source images (single r in this case)

        for box in results[0].boxes.data:
            obj_info = box.cpu().numpy()
            xmin, ymin, xmax, ymax, confidence, class_id = obj_info
            cv2.rectangle(img_disp,
                          np.int16([xmin, ymin]), np.int16([xmax, ymax]),
                          (255, 0, 0), 2)
            ic.putText(img_disp,
                       model.names[int(class_id)] + ' ' + str(confidence),
                       np.int16([xmin, ymin - 5]), (0, 255, 0))


        cv2.imshow('yolo', img_disp)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
