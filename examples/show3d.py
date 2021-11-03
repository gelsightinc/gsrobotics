from matplotlib import pyplot as plt
import sys, getopt
import numpy as np
import cv2
import math
import os
from os import listdir
from os.path import isfile, join
import open3d
import copy
# import utils
# from utils.camera_calibration import warp_perspective
import torch.nn as nn
import torch.nn.functional as F
from gelsight import gsdevice
from gelsight import gs3drecon
from gelsightcore import poisson_reconstruct

def get_diff_img(img1,img2):
    return np.clip((img1.astype(int) - img2.astype(int)), 0, 255).astype(np.uint8)

def get_diff_img_2(img1, img2):
    return (img1 * 1.0 - img2) / 255.  + 0.5

def main(argv):

    device = ""
    try:
       opts, args = getopt.getopt(argv,"hd:",["device="])
    except getopt.GetoptError:
       print('show3d.py -d <device>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('show3d.py -d <device>')
          sys.exit()
       elif opt in ("-d", "--device"):
          device = arg

    # Set flags 
    SAVE_VIDEO_FLAG = False
    GPU = False
    MASK_MARKERS_FLAG = True

    # Path to 3d model
    path = '.'

    # Set the camera resolution
    mmpp = 0.0887  # for 240x320 img size
    # mmpp = 0.1778  # for 160x120 img size from R1
    # mmpp = 0.0446  # for 640x480 img size R1
    # mmpp = 0.029 # for 1032x772 img size from R1

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

    if GPU: device = "cuda"
    else: device = "cpu"
    nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R15)
    net = nn.load_nn(net_path, device)

    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dnnlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (160, 120), isColor=True)

    ''' use this to plot just the 3d '''
    vis3d = gs3drecon.Visualize3D(dev.imgw, dev.imgh, '', mmpp)

    try:
        while dev.while_condition:

            f1 = dev.get_image()

            cv2.imshow('Image', f1)

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
