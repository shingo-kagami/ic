import sys
import cv2
import numpy as np

def byMatchDistance(match):
    return match.distance

def getCenter(topleft, bottomright):
    return ((topleft[0] + bottomright[0]) / 2, (topleft[1] + bottomright[1]) / 2)

def getWidth(topleft, bottomright):
    return bottomright[0] - topleft[0]

def getHeight(topleft, bottomright):
    return bottomright[1] - topleft[1]

def onMouse(event, x, y, flag, state):
    if state["tracking"] == "selecting":
        state["bottomright"] = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        state["tracking"] = "selecting"
        state["topleft"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if (getWidth(state["topleft"], state["bottomright"]) > 0 and
            getHeight(state["topleft"], state["bottomright"]) > 0):
            state["tracking"] = "init_requested"
        else:
            state["tracking"] = "off"


if __name__ == '__main__':
    cap_src = 'vtest.avi'
    if len(sys.argv) == 2:
        if sys.argv[1].isdecimal():
            cap_src = int(sys.argv[1])
        else:
            cap_src = sys.argv[1]
    cap = cv2.VideoCapture(cap_src)

    detector = {
        "SIFT": cv2.xfeatures2d.SIFT_create(), 
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

    method_by_char = {
        "s": "SIFT",
        "b": "BRISK",
        "o": "ORB",
        "k": "KAZE",
        "a": "AKAZE", 
    }
    method = "SIFT"

    for k in method_by_char.keys():
        print(k, ":", method_by_char[k])
    print("drag to set the template")

    state = { "tracking": "off", 
              "topleft": (-1, -1), 
              "bottomright": (-1, -1) }

    cv2.namedWindow("keypoints")
    cv2.setMouseCallback("keypoints", onMouse, state)

    while True:
        retval, img_color = cap.read()
        if retval == False:
            break
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = detector[method].detectAndCompute(img_gray, None)
        for kp in keypoints:
            pt_int = (int(kp.pt[0]), int(kp.pt[1]))
            cv2.circle(img_color, pt_int, 2, (0, 255, 0), -1)

        if state["tracking"] == "init_requested":
            tmplt_center = getCenter(state["topleft"], state["bottomright"])
            twidth = getWidth(state["topleft"], state["bottomright"])
            theight = getHeight(state["topleft"], state["bottomright"])
            tpnts = np.float32([[0, 0],
                                [0, theight -1],
                                [twidth - 1, theight - 1],
                                [twidth - 1, 0]
            ])
            tmplt = cv2.getRectSubPix(img_gray, (twidth, theight), tmplt_center)
            keypoints_tmplt, descriptors_tmplt = detector[method].detectAndCompute(tmplt, None)
            state["tracking"] = "on"

        elif state["tracking"] == "on" and len(keypoints_tmplt) > 0:
            matches = matcher[method].match(descriptors, descriptors_tmplt)
            matches_sorted = sorted(matches, key=byMatchDistance)
            img_matches = cv2.drawMatches(img_gray, keypoints, 
                                          tmplt, keypoints_tmplt, 
                                          matches_sorted[0:10], None)
            cv2.imshow("matches", img_matches)

            if len(matches) >= 4:
                tmplt_points = np.float32([keypoints_tmplt[m.trainIdx].pt 
                                           for m in matches]).reshape(-1, 1, 2)
                image_points = np.float32([keypoints[m.queryIdx].pt 
                                           for m in matches]).reshape(-1, 1, 2)
                G, _ = cv2.findHomography(tmplt_points, image_points, cv2.RANSAC, 5)
                if G is not None:
                    ipnts = cv2.perspectiveTransform(np.float32([tpnts]), G)
                    cv2.polylines(img_color, np.int32(ipnts), 1, (0, 0, 255), 2)

        elif (state["tracking"] == "selecting" and
              getWidth(state["topleft"], state["bottomright"]) > 0 and
              getHeight(state["topleft"], state["bottomright"]) > 0):
            cv2.rectangle(img_color, state["topleft"], state["bottomright"], 
                          (0, 255, 0), 2)

        cv2.putText(img_color, method, (10, 20), cv2.FONT_HERSHEY_PLAIN, 
                    1.0, (255, 255, 0))
        cv2.imshow("keypoints", img_color)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break
        elif key > 0 and chr(key) in method_by_char.keys():
            method = method_by_char[chr(key)]
            state["tracking"] = "off"

    cv2.destroyAllWindows()
