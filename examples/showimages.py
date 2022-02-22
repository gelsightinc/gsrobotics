import sys, getopt
import numpy as np
import cv2
import math
import os
from os import listdir
from os.path import isfile, join
import open3d
import copy
from gelsight import gsdevice

def main(argv):

    device = ""
    try:
       opts, args = getopt.getopt(argv, "hd:", ["device="])
    except getopt.GetoptError:
       print('show3d.py -d <device>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('showimages.py -d <device>')
          print('Use R1 for R1 device, and gsr15???.local for R2 device')
          sys.exit()
       elif opt in ("-d", "--device"):
          device = arg

    # Set flags 
    SAVE_VIDEO_FLAG = False
    GPU = False
    MASK_MARKERS_FLAG = True
    USE_CUSTOM_ROI = False

    # Set the camera resolution
    mmpp = 0.0887  # for 240x320 img size
    # mmpp = 0.1778  # for 160x120 img size from R1
    # mmpp = 0.0446  # for 640x480 img size R1
    # mmpp = 0.029 # for 1032x772 img size from R1

    if device == "R1":
        finger = gsdevice.Finger.R1
    else:
        finger = gsdevice.Finger.R15
        capturestream = "http://" + device + ":8080/?action=stream"

    if finger == gsdevice.Finger.R1:
        dev = gsdevice.Camera(finger, 0)
    else:
        #cap = cv2.VideoCapture('http://gsr15demo.local:8080/?action=stream')
        dev = gsdevice.Camera(finger, capturestream)

    dev.connect()

    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dnnlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (160, 120), isColor=True)

    f0 = dev.get_raw_image()
    print('image size = ', f0.shape[1], f0.shape[0])
    if USE_CUSTOM_ROI:
        roi = cv2.selectROI(f0)
        roi_cropped = f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv2.imshow('ROI', roi_cropped)
        print('Press q in ROI image to continue')
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    elif f0.shape == (640,480,3):
        roi = (60, 100, 375, 380)
    elif f0.shape == (320,240,3):
        roi = (30, 50, 186, 190)
    else:
        roi = (0, 0, f0.shape[1], f0.shape[0])

    print('roi = ', roi)
    print('press q on image to exit')

    try:
        while dev.while_condition:

            # get the roi image
            f1 = dev.get_image(roi)
            bigframe = cv2.resize(f1, (f1.shape[1]*2, f1.shape[0]*2))
            cv2.imshow('Image', bigframe)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if SAVE_VIDEO_FLAG:
                out.write(f1)

    except KeyboardInterrupt:
            print('Interrupted!')
            dev.stop_video()

if __name__ == "__main__":
    main(sys.argv[1:])
