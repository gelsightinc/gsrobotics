import cv2
import numpy as np
import platform
import os
import re

def warp_perspective(img, corners, output_sz):
    TOPLEFT, TOPRIGHT, BOTTOMRIGHT, BOTTOMLEFT = corners
    WARP_H = output_sz[0]
    WARP_W = output_sz[1]
    points1 = np.float32([TOPLEFT, TOPRIGHT, BOTTOMRIGHT, BOTTOMLEFT])
    points2 = np.float32([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    result = cv2.warpPerspective(img, matrix, (WARP_W, WARP_H))
    return result

def get_camera_id(camera_name):
    cam_num = None
    if os.name == 'nt':
        cam_num = find_cameras_windows(camera_name)
    elif platform.system() == "Darwin":
        import usb.core
        devices = usb.core.find(find_all=True)
        for idx, device in enumerate(devices):
            if camera_name in device.product:
                cam_num = idx
                break
    else:
        for file in os.listdir("/sys/class/video4linux"):
            real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
            with open(real_file, "rt") as name_file:
                name = name_file.read().rstrip()
            if camera_name in name:
                cam_num = int(re.search("\d+$", file).group(0))
                found = "FOUND!"
            else:
                found = "      "
            print("{} {} -> {}".format(found, file, name))
    if cam_num is None:
        print("ERROR! Can't Found Camera Device")
        exit()
    return cam_num

if os.name == 'nt':
    def find_cameras_windows(camera_name):

        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()

        # get the device name
        allcams = graph.get_input_devices() # list of camera device
        description = ""
        for cam in allcams:
            if camera_name in cam:
                description = cam
        try:
            device = graph.get_input_devices().index(description)
        except ValueError as e:
            print("Device is not in this list")
            print(graph.get_input_devices())
            import sys
            sys.exit()

        return device


def resize_crop_mini(img, imgw, imgh):
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))
    # keep the ratio the same as the original image size
    img = img[border_size_x+2:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    # final resize for 3d
    img = cv2.resize(img, (imgw, imgh))
    return img


class Camera:
    def __init__(self, dev_type):
        # variable to store data
        self.data = None
        self.name = dev_type
        self.dev_id = get_camera_id(dev_type)
        self.imgw = 320 # this is for R1, R1.5 is 240
        self.imgh = 240 # this is for R1, R1.5 is 320
        self.cam = None
        self.while_condition = 1

    def connect(self):

        # The camera in Mini is a USB camera and uses open cv to get the video data from the streamed video
        self.cam = cv2.VideoCapture(self.dev_id)
        if self.cam is None or not self.cam.isOpened():
            print('Warning: unable to open video source: ', self.dev_id)
        self.imgw = 240
        self.imgh = 320

        return self.cam

    def get_raw_image(self):
        for i in range(10): ## flush out fist 100 frames to remove black frames
            ret, f0 = self.cam.read()
        ret, f0 = self.cam.read()
        if ret:
            f0 = resize_crop_mini(f0,self.imgh,self.imgw)
        else:
            print('ERROR! reading image from camera')

        self.data = f0
        return self.data


    def get_image(self):

        ret, f0 = self.cam.read()
        if ret:
            f0 = resize_crop_mini(f0, self.imgh, self.imgw)
        else:
            print('ERROR! reading image from camera!')

        self.data = f0
        return self.data

    def save_image(self, fname):
         cv2.imwrite(fname, self.data)

    def start_video(self):
        # the default while condition is set to 1, change it for R1.5
        while( self.while_condition ):
            frame = self.get_image()
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

    def stop_video(self):
        cv2.destroyAllWindows()




