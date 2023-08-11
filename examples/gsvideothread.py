import math
import cv2
import time
from threading import Thread
from gelsight import gsdevice

class GsVideo:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        (self.grabbed, self.framei) = self.stream.read()
        self.imgw = 320;
        self.imgh = 240;
        self.frame = gsdevice.resize_crop_mini(self.framei, self.imgw, self.imgh)
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.framei) = self.stream.read()
                self.frame = gsdevice.resize_crop_mini(self.framei, self.imgw, self.imgh)

    def stop(self):
        self.stopped = True

def compute_actual_frame_rate(video_capture, num_frames=500):
    # Initialize variables
    start_time = time.time()
    frame_count = 0
    frame0 = video.frame

    while frame_count < num_frames:
        if (cv2.waitKey(1) == ord("q")) or video.stopped:
            video.stop()
            break

        frame = video.frame
        cv2.imshow("Gelsight Mini", frame)

    # Read frames until reaching the desired number
    #while frame_count < num_frames:
        #ret, frame = video_capture.read()
        #if not ret:
        #    break

        frame_count += 1

    print(f'frame count = {frame_count}')

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate actual frame rate
    actual_frame_rate = frame_count / elapsed_time

    return actual_frame_rate


if __name__ == '__main__':

    # You may need to change vid
    vid = 2
    video = GsVideo(vid)

    video.start()

    frame_rate = video.stream.get(cv2.CAP_PROP_FPS)
    print(f'Camera fps = {frame_rate}')

    fps = frame_rate

    # Compute the actual frame rate
    actual_frame_rate = compute_actual_frame_rate(video)

    # Display the actual frame rate
    print("Actual frame rate: {:.2f} FPS".format(actual_frame_rate))

    # used to record the time when we processed last frame
    prev_frame_time = 0.

    # used to record the time at which we processed current frame
    new_frame_time = 0.

    dt = 1/fps
    ptime = 0.
    dfps = 0

    while True:
        if (cv2.waitKey(1) == ord("q")) or video.stopped:
            video.stop()
            break

        # Calculating the fps
        new_frame_time = time.time()

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result

        if prev_frame_time > 0:
            ptime = ptime + (new_frame_time - prev_frame_time)

        if ptime > dt:

            frame = video.frame

            actual = 1./ptime
            print('fps = %.5f' % actual)

            # converting the fps into integer
            if ptime > 0:
                dfps = math.ceil(actual)

            # converting the fps to string so that we can display it on frame
            # by using putText function
            dfps = str(dfps)

            # putting the FPS coun=curr_timet on the frame
            cv2.putText(frame, dfps, (7, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

            cv2.imshow("Gelsight Mini", frame)
            ptime = 0

        prev_frame_time = new_frame_time

    cv2.destroyAllWindows()