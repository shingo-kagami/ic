import sys
import cv2
import numpy as np

if __name__ == '__main__':
    cv2.namedWindow('disp')

    cap_src = 'vtest.avi'
    if len(sys.argv) == 2:
        if sys.argv[1].isdecimal():
            cap_src = int(sys.argv[1])
        else:
            cap_src = sys.argv[1]
    cap = cv2.VideoCapture(cap_src)

    track_mode = True
    need_init = True
    blackout = False
    flow_at_grid = False

    max_points = 1024
    lk_params = dict(winSize = (21, 21),
                     maxLevel = 5,
                     criteria = (cv2.TERM_CRITERIA_EPS 
                                 | cv2.TERM_CRITERIA_COUNT, 
                                 10, 0.03))

    retval, input_prev = cap.read()
    input_prev = cv2.cvtColor(input_prev, cv2.COLOR_BGR2GRAY)

    print("t: toggle between tracking and flow modes");
    print("b: toggle blackout");
    print("g: toggle between grid and feature points (flow mode)");
    print("i: initialize feature points (track mode)");
    print("s: save an image");

    while True:
        retval, input = cap.read()
        input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

        if (not track_mode) and flow_at_grid:
            points_prev = np.array([ [[x, y]] 
                                     for x in range(10, input.shape[1], 10) 
                                     for y in range(10, input.shape[0], 10) ], 
                                   dtype=np.float32)
        elif (not track_mode) or need_init:
            points_prev = cv2.goodFeaturesToTrack(input_prev, max_points, 
                                                  0.001, 10)
            need_init = False
        else:
            points_prev = points

        points, status, err = cv2.calcOpticalFlowPyrLK(input_prev, input, 
                                                       points_prev, None, 
                                                       **lk_params)
        input_prev = input.copy()

        disp = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        if blackout:
            disp = np.zeros_like(disp)
        for i in range(len(points)):
            if status[i] == 0:
                continue
            ipnt_prev = int(points_prev[i][0][0]), int(points_prev[i][0][1])
            ipnt = int(points[i][0][0]), int(points[i][0][1])

            if not track_mode:
                cv2.line(disp, ipnt_prev, ipnt, (0, 255, 0), 2)
            else:
                cv2.circle(disp, ipnt, 2, (0, 255, 0), -1)

        cv2.imshow("disp", disp)
        key = cv2.waitKey(30)

        if key == ord('q'):
            break
        elif key == ord('t'):
            track_mode = not track_mode
            if track_mode: need_init = True
        elif key == ord('b'):
            blackout = not blackout
        elif key == ord('g'):
            flow_at_grid = not flow_at_grid
        elif key == ord('i'):
            need_init = True
        elif key == ord('s'):
            cv2.imwrite('klt_flow.jpg', disp)

    cap.release()
    cv2.destroyAllWindows()
