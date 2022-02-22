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
from gelsight import gs3drecon
from gelsightcore import poisson_reconstruct

def get_diff_img(img1, img2):
    return np.clip((img1.astype(int) - img2.astype(int)), 0, 255).astype(np.uint8)

def get_diff_img_2(img1, img2):
    return (img1 * 1.0 - img2) / 255.  + 0.5

def main(argv):

    device = ""
    try:
       opts, args = getopt.getopt(argv, "hd:", ["device="])
    except getopt.GetoptError:
       print('show3d.py -d <device>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('show3d.py -d <device>')
          print('Use R1 for R1 device, and gsr15???.local for R2 device')
          sys.exit()
       elif opt in ("-d", "--device"):
          device = arg

    # Set flags 
    SAVE_VIDEO_FLAG = False
    GPU = False
    MASK_MARKERS_FLAG = True
    USE_CUSTOM_ROI = False

    # Path to 3d model
    path = '.'

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
        net_file_path = 'nnr1.pt'
    else:
        #cap = cv2.VideoCapture('http://gsr15demo.local:8080/?action=stream')
        dev = gsdevice.Camera(finger, capturestream)
        net_file_path = 'nnr15.pt'

    dev.connect()

    ''' Load neural network '''
    model_file_path = path
    net_path = os.path.join(model_file_path, net_file_path)
    print('net path = ', net_path)

    if GPU: gpuorcpu = "cuda"
    else: gpuorcpu = "cpu"
    if device=="R1":
        nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R1)
    else:
        nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R15)
    net = nn.load_nn(net_path, gpuorcpu)

    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dnnlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (160, 120), isColor=True)

    f0 = dev.get_raw_image()
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
    ''' use this to plot just the 3d '''
    vis3d = gs3drecon.Visualize3D(dev.imgw, dev.imgh, '', mmpp)

    try:
        while dev.while_condition:

            # get the roi image
            f1 = dev.get_image(roi)
            bigframe = cv2.resize(f1, (f1.shape[1]*2, f1.shape[0]*2))
            cv2.imshow('Image', bigframe)

            # compute the depth map
            dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG)

            ''' Display the results '''
            vis3d.update(dm)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if SAVE_VIDEO_FLAG:
                out.write(f1)

    except KeyboardInterrupt:
            print('Interrupted!')
            dev.stop_video()

if __name__ == "__main__":
    main(sys.argv[1:])
