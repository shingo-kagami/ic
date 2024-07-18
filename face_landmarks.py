import sys
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

import ic_utils as ic


def draw_face_mesh(img, face_landmarks):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
        for lm in face_landmarks])
    solutions.drawing_utils.draw_landmarks(
      img_rgb,
      face_landmarks_proto,
      solutions.face_mesh.FACEMESH_TESSELATION,
      None,
      solutions.drawing_styles.get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
      img_rgb,
      face_landmarks_proto,
      solutions.face_mesh.FACEMESH_CONTOURS,
      None,
      solutions.drawing_styles.get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
      img_rgb,
      face_landmarks_proto,
      solutions.face_mesh.FACEMESH_IRISES,
      None,
      solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def main():
    cap = ic.select_capture_source(sys.argv)

    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           num_faces=2)
    detector = vision.FaceLandmarker.create_from_options(options)

    cv2.namedWindow('result')
    cv2.createTrackbar('highlight', 'result', 0, 477, ic.do_nothing)

    while True:
        grabbed, img = cap.read()
        if not grabbed:
            break

        height = img.shape[0]
        width = img.shape[1]
        img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        results = detector.detect(img_mp)

        highlight_index = cv2.getTrackbarPos('highlight', 'result')

        num_detected_faces = len(results.face_landmarks)

        for k in range(num_detected_faces):
            img = draw_face_mesh(img, results.face_landmarks[k])

            lm = results.face_landmarks[k][highlight_index]
            x = int(lm.x * width)
            y = int(lm.y * height)
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        cv2.imshow('result', img)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
