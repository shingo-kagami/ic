import sys
import cv2
import dlib

import ic_utils as ic


def main():
    face_model_file = 'shape_predictor_68_face_landmarks.dat'
    #face_model_file = 'shape_predictor_5_face_landmarks.dat'
    upsampling_enabled = 1

    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(face_model_file)

    cap = ic.select_capture_source(sys.argv)

    while True:
        grabbed, img_color = cap.read()
        if not grabbed:
            break
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        face_rectangles = face_detector(img_gray, upsampling_enabled)

        for rect in face_rectangles:
            # draw detected face rectangle
            x0, y0 = rect.tl_corner().x, rect.tl_corner().y
            x1, y1 = rect.br_corner().x, rect.br_corner().y
            cv2.rectangle(img_color, (x0, y0), (x1, y1), (255, 0, 0), 2)

            # draw face landmarks
            shape_obj = face_predictor(img_gray, rect)
            landmarks = shape_obj.parts()
            for lm in landmarks:
                cv2.circle(img_color, (lm.x, lm.y), 2, (0, 255, 0), -1)

        cv2.imshow('face landmarks', img_color)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
