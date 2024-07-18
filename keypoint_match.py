import sys
import cv2
import numpy as np

import ic_utils as ic


def byMatchDistance(match):
    return match.distance


def main():
    cap = ic.select_capture_source(sys.argv)

    method_list = [
        "SIFT",
        "BRISK",
        "ORB",
        "KAZE",
        "AKAZE", 
    ]

    detector = {
        "SIFT": cv2.SIFT_create(), 
        "KAZE": cv2.KAZE_create(), 
        "BRISK": cv2.BRISK_create(), 
        "ORB": cv2.ORB_create(nfeatures=3000), 
        "AKAZE": cv2.AKAZE_create(), 
    }

    matcher_l2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matcher_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matcher = {
        "SIFT": matcher_l2, 
        "KAZE": matcher_l2, 
        "BRISK": matcher_hamming, 
        "ORB": matcher_hamming, 
        "AKAZE": matcher_hamming, 
    }

    mstate = {
        'selection': 'invalid',
        'xybegin': (-1, -1),
        'xyend': (-1, -1),
    }

    cv2.namedWindow('keypoints')
    cv2.setMouseCallback('keypoints', ic.on_mouse_rect, mstate)
    cv2.createTrackbar('methods', 'keypoints', 0, len(method_list) - 1,
                       ic.do_nothing)

    target = None
    method_target = None

    while True:
        grabbed, frame_color = cap.read()
        if not grabbed:
            break
        frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

        method_id = cv2.getTrackbarPos('methods', 'keypoints')
        method = method_list[method_id]

        keypoints, descriptors = detector[method].detectAndCompute(frame, None)
        for kp in keypoints:
            cv2.circle(frame_color, np.int16(kp.pt), 2, (0, 255, 0), -1)

        if mstate['selection'] == 'valid' and target is None:
            ## initialization requested

            method_target = method
            xybegin = np.array(mstate['xybegin'])
            xyend = np.array(mstate['xyend'])
            xycenter = np.int16((xybegin + xyend) / 2)

            tsize = xyend - xybegin   ## == (width, height)
            target = cv2.getRectSubPix(frame, tsize, xycenter)

            tpnts = np.float32([[0, 0],
                                [0, tsize[1]],
                                [tsize[0], tsize[1]],
                                [tsize[0], 0]
                                ])

            keypoints_target, descriptors_target \
                = detector[method].detectAndCompute(target, None)


        if (mstate['selection'] == 'valid' and len(keypoints_target) > 0
            and method == method_target):
            matches = matcher[method].match(descriptors, descriptors_target)
            matches_sorted = sorted(matches, key=byMatchDistance)
            img_matches = cv2.drawMatches(frame, keypoints,
                                          target, keypoints_target, 
                                          matches_sorted[0:10], None)

            cv2.imshow('matches', img_matches)

            if len(matches) >= 4:
                target_points = np.float32([keypoints_target[m.trainIdx].pt 
                                            for m in matches]).reshape(-1, 1, 2)
                image_points = np.float32([keypoints[m.queryIdx].pt 
                                           for m in matches]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(target_points, image_points,
                                          cv2.RANSAC, 5)
                if H is not None:
                    ipnts = cv2.perspectiveTransform(np.float32([tpnts]), H)
                    cv2.polylines(frame_color, np.int32(ipnts), 1,
                                  (0, 0, 255), 2)

        elif mstate['selection'] == 'ongoing':
            target = None

            xybegin = np.array(mstate['xybegin'])
            xyend = np.array(mstate['xyend'])
            cv2.rectangle(frame_color, xybegin, xyend, (255, 0, 0), 2)

        elif mstate['selection'] == 'invalid':
            target = None


        ic.putText(frame_color, method, (10, 20), (255, 255, 0))
        cv2.imshow('keypoints', frame_color)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
