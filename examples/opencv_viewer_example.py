###############################################
##      Open CV and Numpy integration        ##
###############################################

from gelsight import gsdevice
from gelsight import gs3drecon
import numpy as np
import cv2

# Configure depth and color streams
finger = gsdevice.Finger.R1

if finger == gsdevice.Finger.R1:
    dev = gsdevice.Camera(finger, 0)
    net_file_path = 'nnr1.pt'
else:
    #cap = cv2.VideoCapture('http://gsr15demo.local:8080/?action=stream')
    dev = gsdevice.Camera(finger, 'http://gsr15demo.local:8080/?action=stream')
    net_file_path = 'nnr15.pt'

dev.connect()

#dev.set_size(dev.imgh*2, dev.imgw*2)

# Load the model for 3d reconstruction
nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R1)
net = nn.load_nn(net_file_path, 'cpu')

# Start streaming

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        color_image = dev.get_image()
        depth_image = nn.get_depthmap(color_image, 'True')

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('Gelsight', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Gelsight', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    dev.stop_video()
