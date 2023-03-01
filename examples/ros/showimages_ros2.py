import cv2
from PIL import Image
from threading import Thread, Lock
import rclpy
from rclpy.node import Node  # Enables the use of rclpy's Node class
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class CameraPublisher(Node):
    """
    Create a CameraPublisher class, which is a subclass of the Node class.
    The class publishes the position of an object every 3 seconds.
    The position of the object are the x and y coordinates with respect to
    the camera frame.
    """

    def __init__(self):
        """
        Class constructor to set up the node
        """

        # Initiate the Node class's constructor and give it a name
        super().__init__('camera_publisher')

        # Create publisher(s)

        # This node publishes the position of an object every 3 seconds.
        # Maximum queue size of 10.
        self.publisher_position_cam_frame = self.create_publisher(Image, '/gsmini_rawimg_0', 10)

        # 3 seconds
        timer_period = 0.05
        self.timer = self.create_timer(timer_period, self.get_image)
        self.i = 0  # Initialize a counter variable

        self.vs0 = WebcamVideoStream(src=2).start()
        self.cvbridge = CvBridge()

    def get_image(self):
        """
        Callback function.
        This function gets called every 3 seconds.
        We locate an object using the camera and then publish its coordinates to ROS2 topics.
        """
        img = self.vs0.read()
        # Publish the coordinates to the topic
        self.publish_coordinates(img)

        # Increment counter variable
        self.i += 1

    def publish_coordinates(self, img):
        """
        Publish the coordinates of the object to ROS2 topics
        :param: The position of the object in centimeter coordinates [x , y]
        """
        # msg = Image()  # Create a message of this type
        msg = self.cvbridge.cv2_to_imgmsg(img, encoding="passthrough")  # Store the x and y coordinates of the object
        self.publisher_position_cam_frame.publish(msg)  # Publish the position to the topic



def get_diff_img_2(img1, img2):
    return (img1 * 1.0 - img2) / 255.  +  0.5


class WebcamVideoStream :
    def __init__(self, src, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()


def chop_border_resize(img):
    img = img[chop_border_size:imgh - chop_border_size, chop_border_size:imgw - chop_border_size]
    img = cv2.resize(img, (imgw, imgh))
    return img


def main(args=None):


    SAVE_VIDEO_FLAG = False
    SAVE_SINGLE_IMGS_FLAG = False
    cvbridge = CvBridge()
    chop_border_size = 0
    imgh = 240
    imgw = 320
    NUM_SENSORS = 1

    gs = {}
    gs['img'] = [0] * 2
    gs['gsmini_pub'] = [0] * 2
    gs['vs'] = [0] * 2
    gs['img_msg'] = [0] * 2

    # for i in range(NUM_SENSORS):
    #     # gs['gsmini_pub'][i] = rospy.Publisher("/gsmini_rawimg_{}".format(i), Image, queue_size=1)
    #     gs['gsmini_pub'][i] = create_publisher(Image, '/pos_in_cam_frame', 10)
    #     gs['vs'][i] = WebcamVideoStream(src=2*i + 2).start() # make sure the id numbers of the cameras are ones recognized by the computer. Default, 2 and 4
    #
    # r = rospy.Rate(25)  # 10hz
    # while not rospy.is_shutdown():
    #
    #     for i in range(NUM_SENSORS):
    #         gs['img'][i] = gs['vs'][i].read()
    #         gs['img'][i] = cv2.resize(gs['img'][i], (imgw,imgh))
    #         gs['img'][i] = chop_border_resize(gs['img'][i])
    #         cv2.imshow('gsmini{}'.format(i), gs['img'][i])
    #
    #     print ('.. hit ESC to exit! .. ')
    #     if cv2.waitKey(1) == 27 :
    #         break
    #
    #     ''' publish imgae to ros '''
    #     for i in range(NUM_SENSORS):
    #         gs['img_msg'][i] = cvbridge.cv2_to_imgmsg(gs['img'][i], encoding="passthrough")
    #         gs['img_msg'][i].header.stamp = rospy.Time.now()
    #         gs['img_msg'][i].header.frame_id = 'map'
    #         gs['gsmini_pub'][i].publish(gs['img_msg'][i])
    #
    #     r.sleep()
    #
    #
    # for i in range(NUM_SENSORS): gs['vs'][i].stop()
    # cv2.destroyAllWindows()

    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    camera_publisher = CameraPublisher()

    # Spin the node so the callback function is called.
    # Publish any pending messages to the topics.
    rclpy.spin(camera_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    camera_publisher.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()



if __name__ == '__main__':
    main()

