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
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg



def get_diff_img(img1, img2):
    return np.clip((img1.astype(int) - img2.astype(int)), 0, 255).astype(np.uint8)


def get_diff_img_2(img1, img2):
    return (img1 * 1.0 - img2) / 255. + 0.5


class PCDPublisher(Node):

    def __init__(self):
        super().__init__('pcd_publisher_node')

        device = ""
        try:
            opts, args = getopt.getopt(sys.argv, "hd:", ["device="])

            print ('\n\n\nargs[2]:', args, '\n\n\n')

        except getopt.GetoptError:
            print('python show3d.py -d <device>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print('show3d.py -d <device>')
                print('Use R1 for R1 device, and gsr15???.local for R2 device')
                sys.exit()
            elif opt in ("-d", "--device"):
                device = sys.argv[1]


        device = sys.argv[2]


        # Set flags
        SAVE_VIDEO_FLAG = False
        GPU = False
        MASK_MARKERS_FLAG = True
        FIND_ROI = False
        PUBLISH_ROS_PC = True

        # Path to 3d model
        path = '.'

        # Set the camera resolution
        # mmpp = 0.0887  # for 240x320 img size
        # mmpp = 0.1778  # for 160x120 img size from R1
        # mmpp = 0.0446  # for 640x480 img size R1
        # mmpp = 0.029 # for 1032x772 img size from R1
        mmpp = 0.081  # r2d2 gel 18x24mm at 240x320
        self.mpp = mmpp / 1000.

        print ('\n\n\n', device, '\n\n\n')

        if device == "R1":
            finger = gsdevice.Finger.R1
        elif device[-5:] == "local":
            finger = gsdevice.Finger.R15
            capturestream = "http://" + device + ":8080/?action=stream"
        elif device == "mini":
            finger = gsdevice.Finger.MINI
        else:
            print('Unknown device name')
            print('Use R1 for R1 device \ngsr15???.local for R1.5 device \nmini for mini device')

        if finger == gsdevice.Finger.R1:
            self.dev = gsdevice.Camera(finger, 0)
            net_file_path = 'nnr1.pt'
        elif finger == gsdevice.Finger.R15:
            # cap = cv2.VideoCapture('http://gsr15demo.local:8080/?action=stream')
            self.dev = gsdevice.Camera(finger, capturestream)
            net_file_path = 'nnr15.pt'
        elif finger == gsdevice.Finger.MINI:
            # the device ID can change after chaning the usb ports.
            # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
            cam_id = gsdevice.get_camera_id("Arducam USB Camera")
            self.dev = gsdevice.Camera(finger, cam_id)
            net_file_path = '../nnmini.pt'

        self.dev.connect()

        ''' Load neural network '''
        model_file_path = path
        net_path = os.path.join(model_file_path, net_file_path)
        print('net path = ', net_path)

        if GPU:
            gpuorcpu = "cuda"
        else:
            gpuorcpu = "cpu"
        if device == "R1":
            self.nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R1, self.dev)
        else:
            self.nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R15, self.dev)
        net = self.nn.load_nn(net_path, gpuorcpu)

        if SAVE_VIDEO_FLAG:
            #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
            file_path = './3dnnlive.mov'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(file_path, fourcc, 60, (160, 120), isColor=True)

        f0 = self.dev.get_raw_image()

        if FIND_ROI:
            self.roi = cv2.selectROI(f0)
            self.roi_cropped = f0[int(self.roi[1]):int(self.roi[1] + self.roi[3]), int(self.roi[0]):int(self.roi[0] + self.roi[2])]
            cv2.imshow('ROI', self.roi_cropped)
            print('Press q in ROI image to continue')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif f0.shape == (640, 480, 3):
            self.roi = (60, 100, 375, 380)
        elif f0.shape == (320, 240, 3):
            self.roi = (30, 50, 186, 190)
        elif f0.shape == (240, 320, 3):
            ''' cropping is hard coded in resize_crop_mini() function in gsdevice.py file '''
            border_size = 0  # default values set for mini to get 3d
            self.roi = (
                border_size, border_size, 320 - 2 * border_size,
                240 - 2 * border_size)  # default values set for mini to get 3d
        else:
            self.roi = (0, 0, f0.shape[1], f0.shape[0])

        print('roi = ', self.roi)
        print('press q on image to exit')


        ''' point array to store point cloud data points '''
        x = np.arange(self.dev.imgh) * self.mpp
        y = np.arange(self.dev.imgw) * self.mpp
        X, Y = np.meshgrid(x, y)
        self.points = np.zeros([self.dev.imgw * self.dev.imgh, 3])
        self.points[:, 0] = np.ndarray.flatten(X)
        self.points[:, 1] = np.ndarray.flatten(Y)
        Z = np.zeros((self.dev.imgh, self.dev.imgw))  # initialize points array with zero depth values
        self.points[:, 2] = np.ndarray.flatten(Z)

        # I create a publisher that publishes sensor_msgs.PointCloud2 to the
        # topic 'pcd'. The value '10' refers to the history_depth, which I
        # believe is related to the ROS1 concept of queue size.
        # Read more here:
        # http://wiki.ros.org/rospy/Overview/Publishers%20and%20Subscribers
        self.pcd_publisher = self.create_publisher(PointCloud2, 'pcd', 10)
        timer_period = 1 / 25.0
        self.timer = self.create_timer(timer_period, self.timer_callback)


    def timer_callback(self):

        # get the roi image
        f1 = self.dev.get_image(self.roi)

        # compute the depth map
        dm = self.nn.get_depthmap(f1, False)

        dm_ros = copy.deepcopy(dm) * self.mpp
        self.points[:, 2] = np.ndarray.flatten(dm_ros)

        # Here I use the point_cloud() function to convert the numpy array
        # into a sensor_msgs.PointCloud2 object. The second argument is the
        # name of the frame the point cloud will be represented in. The default
        # (fixed) frame in RViz is called 'map'
        self.pcd = point_cloud(self.points, 'map')
        # Then I publish the PointCloud2 object
        self.pcd_publisher.publish(self.pcd)


def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx3 array of xyz positions.
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    Code source:
        https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
    References:
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointCloud2.html
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html
        http://docs.ros.org/melodic/api/std_msgs/html/msg/Header.html
    """
    # In a PointCloud2 message, the point cloud is stored as an byte
    # array. In order to unpack it, we also include some parameters
    # which desribes the size of each individual point.
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.

    data = points.astype(dtype).tobytes()

    # The fields specify what the bytes represents. The first 4 bytes
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [PointField(
        name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyz')]

    # The PointCloud2 message also has a header which specifies which
    # coordinate frame it is represented in.
    header = std_msgs.msg.Header(frame_id=parent_frame)

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 3),  # Every point consists of three float32s.
        row_step=(itemsize * 3 * points.shape[0]),
        data=data
    )


def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)

    pcd_publisher = PCDPublisher()
    rclpy.spin(pcd_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pcd_publisher.destroy_node()
    rclpy.shutdown()



if __name__ == "__main__":
    main()
