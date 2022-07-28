import sys
import cv2
import numpy as np
import torch

import ic_utils as ic


def main():
    torch.hub.set_dir('./hub_cache')
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

    model.cpu()
    # model.cuda()

    class_names = model.names
    """
    print(class_names)
    => [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    """

    cap = ic.select_capture_source(sys.argv)

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        img_disp = frame.copy()
        results = model(frame)

        for detection in results.xyxy[0]:
            obj_info = detection.cpu().numpy()
            xmin, ymin, xmax, ymax, confidence, class_id = obj_info

            cv2.rectangle(img_disp,
                          np.int16([xmin, ymin]), np.int16([xmax, ymax]),
                          (255, 0, 0), 2)
            ic.putText(img_disp,
                       class_names[int(class_id)] + ' ' + str(confidence),
                       np.int16([xmin, ymin - 5]), (255, 255, 255))

        t_preprocess, t_infer, t_nonmax_suppress = results.t
        t_total = np.sum(results.t)
        time_stat_msg = 'total: {:.1f} ms, inference: {:.1f} ms'.format(
            t_total, t_infer
        )
        ic.putText(img_disp, time_stat_msg, (10, 20), (0, 255, 0))

        results.render()
        cv2.imshow('rendered', frame) ## frame was modified by results.render()
        cv2.imshow('result', img_disp)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
