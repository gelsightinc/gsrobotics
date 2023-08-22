import cv2
import numpy as np
from threading import Thread, Lock
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def flat_field_correction(f0, img, counter, num_init_frames, KJ):

    h,w = img.shape[0], img.shape[1]

    mean = np.mean(f0, axis=0)
    J = img / mean

    # imgffc[:,:,0] = (imgffc[:,:,0] - imgffc[:,:,0].min())/ (imgffc[:,:,0].max() - imgffc[:,:,0].min())
    # imgffc[:, :, 1] = (imgffc[:, :, 1] - imgffc[:, :, 1].min()) / (imgffc[:, :, 1].max() - imgffc[:, :, 1].min())
    # imgffc[:, :, 2] = (imgffc[:, :, 2] - imgffc[:, :, 2].min()) / (imgffc[:, :, 2].max() - imgffc[:, :, 2].min())

    J_min = np.tile(np.min(np.min(J, axis=1), axis=0).reshape(1,-1),(h,w,1))
    J_max = np.tile(np.max(np.max(J, axis=1), axis=0).reshape(1,-1),(h,w,1))

    R = (J - J_min) / (J_max - J_min)
    aa = np.sum(np.sum(J, axis=1), axis=0) / np.sum(np.sum(R, axis=1), axis=0)
    K = np.tile(aa.reshape(1,-1),(h,w,1))
    J = K*R

    if counter==num_init_frames:
        KJ = np.round(np.mean(255 / np.max(np.max(J, axis=1), axis=0).reshape(1, -1)))

    J = KJ * J
    J[J<0] = 0
    J[J>255] = 255

    return J.astype('uint8'), KJ



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



if __name__ == '__main__':

    rospy.init_node('twogsminiros', anonymous=True)

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

    for i in range(NUM_SENSORS):
        gs['gsmini_pub'][i] = rospy.Publisher("/gsmini_rawimg_{}".format(i), Image, queue_size=1)
        gs['vs'][i] = WebcamVideoStream(src=2*i + 2).start() # make sure the id numbers of the cameras are ones recognized by the computer. Default, 2 and 4

    r = rospy.Rate(25)  # 10hz
    while not rospy.is_shutdown():

        for i in range(NUM_SENSORS):
            gs['img'][i] = gs['vs'][i].read()
            gs['img'][i] = cv2.resize(gs['img'][i], (imgw,imgh))
            #gs['img'][i] = chop_border_resize(gs['img'][i])
            cv2.imshow('gsmini{}'.format(i), gs['img'][i])

        print ('.. hit ESC to exit! .. ')
        if cv2.waitKey(1) == 27 :
            break

        ''' publish image to ros '''
        for i in range(NUM_SENSORS):
            gs['img_msg'][i] = cvbridge.cv2_to_imgmsg(gs['img'][i], encoding="passthrough")
            gs['img_msg'][i].header.stamp = rospy.Time.now()
            gs['img_msg'][i].header.frame_id = 'map'
            gs['gsmini_pub'][i].publish(gs['img_msg'][i])

        r.sleep()

    for i in range(NUM_SENSORS): gs['vs'][i].stop()
    cv2.destroyAllWindows()

