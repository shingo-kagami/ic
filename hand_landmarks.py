import sys
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

import ic_utils as ic


def draw_hand(img, landmarks):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
        for lm in landmarks])
    solutions.drawing_utils.draw_landmarks(
      img_rgb,
      landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def main():
    cap = ic.select_capture_source(sys.argv)

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=2,
                                           min_hand_detection_confidence=0.5,
                                           min_hand_presence_confidence=0.5,
                                           min_tracking_confidence=0.5,
                                           )
    detector = vision.HandLandmarker.create_from_options(options)


    cv2.namedWindow('result')
    cv2.createTrackbar('highlight', 'result', 0, 20, ic.do_nothing)

    while True:
        grabbed, img = cap.read()
        if not grabbed:
            break

        height = img.shape[0]
        width = img.shape[1]
        img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        results = detector.detect(img_mp)

        highlight_index = cv2.getTrackbarPos('highlight', 'result')
        num_detected_hands = len(results.hand_landmarks)

        for k in range(num_detected_hands):
            img = draw_hand(img, results.hand_landmarks[k])

            handedness = results.handedness[k][0].index
            if handedness == 0: # right hand -> red
                color = (0, 0, 255)
            else: # left hand -> cyan
                color = (255, 255, 0)

            lm = results.hand_landmarks[k][highlight_index]
            x = int(lm.x * width)
            y = int(lm.y * height)
            cv2.circle(img, (x, y), 10, color, 2)

        cv2.imshow('result', img)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
