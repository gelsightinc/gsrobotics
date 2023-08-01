#!usr/bin/python3.8

import cv2
import numpy as np
from gelsight import gsdevice
from markertracker import MarkerTracker

if __name__ == "__main__":

    imgw = 320
    imgh = 240
    SAVE_VIDEO_FLAG = True
    USE_MINI_LIVE = True

    DRAW_MARKERS = True
    if SAVE_VIDEO_FLAG:
        DRAW_MARKERS = False

    if USE_MINI_LIVE:
        gs = gsdevice.Camera("GelSight Mini")
        WHILE_COND = 1
        gs.connect()
    else:
        cap = cv2.VideoCapture('data/mini_example.avi') # choose to read from video or camera
        WHILE_COND = cap.isOpened()

    if SAVE_VIDEO_FLAG:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('flow.mov',fourcc, 60.0, (imgw,imgh)) # The fps depends on CPU

    if USE_MINI_LIVE:
        f0 = gs.get_raw_image()
        #f0 = cv2.resize(f0, (imgw, imgh))
        f0gray = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    else:
        ret, f0 = cap.read()
        f0gray = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)

    img = np.float32(f0) / 255.0
    mtracker = MarkerTracker(img)

    marker_centers = mtracker.initial_marker_center
    Ox = marker_centers[:, 1]
    Oy = marker_centers[:, 0]
    nct = len(marker_centers)

    # Convert the first frame to grayscale
    old_gray = f0gray

    # Set the parameters for the Lucas-Kanade method
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors for visualizing the optical flow
    color = np.random.randint(0, 255, (100, 3))

    # Existing p0 array
    p0 = np.array([[Ox[0], Oy[0]]], np.float32).reshape(-1, 1, 2)
    for i in range(nct - 1):
        # New point to be added
        new_point = np.array([[Ox[i+1], Oy[i+1]]], np.float32).reshape(-1, 1, 2)
        # Append new point to p0
        p0 = np.append(p0, new_point, axis=0)

    try:
        while (WHILE_COND):

            if USE_MINI_LIVE:
                curr = gs.get_image()
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            else:
                ret, curr = cap.read()
                if ret:
                    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                else:
                    break

            # Calculate the optical flow using the Lucas-Kanade method
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, curr_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) < nct:
                # Detect new features in the current frame
                print(f"all pts did not converge")
            else:
                # Update points for next iteration
                p0 = good_new.reshape(-1, 1, 2)

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                ix = int(Ox[i])
                iy = int(Oy[i])
                offrame = cv2.arrowedLine(curr, (ix,iy), (int(a), int(b)), (255,255,255), thickness=1, line_type=cv2.LINE_8, tipLength=.15)
                #offrame = cv2.line(curr, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                if DRAW_MARKERS:
                    offrame = cv2.circle(offrame, (int(a), int(b)), 5, color[i].tolist(), -1)

            # Show the video with the optical flow tracks overlaid
            cv2.imshow('optical flow frame', cv2.resize(offrame, (2*offrame.shape[1], 2*offrame.shape[0])))
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            # Update the previous frame and points
            old_gray = curr_gray.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if SAVE_VIDEO_FLAG:
                out.write(offrame)

        if USE_MINI_LIVE:
            gs.stop_video()
        else:
            cap.release()
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
            print('Interrupted!')
            if USE_MINI_LIVE:
                gs.stop_video()
            else:
                cap.release()
                cv2.destroyAllWindows()
