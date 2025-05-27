import cv2
import numpy as np
import datetime
from utilities.gelsightmini import GelSightMini

FULLWIDTH = 3280
FULLHEIGHT= 2464

# Image size and crop
imgw = FULLWIDTH
imgh = FULLHEIGHT
brd_frac = 0

imgw = 640 # FULLWIDTH
imgh = 480 # FULLHEIGHT
brd_frac = 0.15


def show_image():
    counter = 0
    cam = GelSightMini(
        target_width=imgw,
        target_height=imgh,
        border_fraction=brd_frac)

    deviceidx = cam.select_device()

    start = datetime.datetime.now()

    # Settings for FPS overlay
    org = (50, 50) # Bottom-left corner of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255) # White color in BGR
    thickness = 2
    nframeavg = 10
    fpsstr = str(0)

    cam.start()
    while True:
        if counter % nframeavg == 0:
            frame_start = datetime.datetime.now()

        img = cam.update(1.0)

        if img.shape[0] > 0:
            counter = counter + 1

            if counter >= nframeavg and counter % nframeavg == 0:
                frame_end = datetime.datetime.now()
                frame_time = frame_end - frame_start
                fpstemp = nframeavg / frame_time.total_seconds()
                fpsstr = f"{fpstemp:.1f} FPS"

            # imshow assumes BGR order
            imbgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.putText(imbgr, fpsstr, org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('frame_rgb', imbgr)

        if cv2.waitKey(1) == ord('q'):
            end = datetime.datetime.now()
            total_time = end - start
            avg_time = total_time.total_seconds() / counter * 1000
            fps = 1000 / avg_time
            print(avg_time, fps)
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':

    show_image()
