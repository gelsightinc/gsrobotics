import sys, getopt
import cv2
from gelsight import gsdevice


def main(argv):

    device = "mini"
    try:
       opts, args = getopt.getopt(argv, "hd:", ["device="])
    except getopt.GetoptError:
       print('python showimages.py -d <device>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('showimages.py -d <device>')
          print('Use R1 for R1 device \ngsr15???.local for R1.5 device \nmini for mini device')
          sys.exit()
       elif opt in ("-d", "--device"):
          device = arg

    # Set flags 
    SAVE_VIDEO_FLAG = False
    GPU = False
    MASK_MARKERS_FLAG = False
    FIND_ROI = False ## set True the first time to find ROI on an interactive window. Once values are found, use one of the else block.

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
        dev = gsdevice.Camera(finger, 0)
    elif finger == gsdevice.Finger.R15:
        #cap = cv2.VideoCapture('http://gsr15demo.local:8080/?action=stream')
        dev = gsdevice.Camera(finger, capturestream)
    elif finger == gsdevice.Finger.MINI:
        # the device ID can change after unplugging and changing the usb ports.
        # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
        cam_id = gsdevice.get_camera_id("GelSight Mini")
        dev = gsdevice.Camera(finger, cam_id)

    dev.connect()

    f0 = dev.get_raw_image()
    print('image size = ', f0.shape[1], f0.shape[0])
    roi = (0, 0, f0.shape[1], f0.shape[0])

    if FIND_ROI:
        roi = cv2.selectROI(f0)
        roi_cropped = f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv2.imshow('ROI', roi_cropped)
        print('Press q in ROI image to continue')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif f0.shape == (640,480,3) and device != 'mini':
        roi = (60, 100, 375, 380)
    elif f0.shape == (320,240,3) and device != 'mini':
        roi = (30, 50, 186, 190)
    elif f0.shape == (240,320,3) and device != 'mini':
        ''' cropping is hard coded in resize_crop_mini() function in gsdevice.py file '''
        border_size = 0 # default values set for mini to get 3d
        roi = (border_size,border_size,320-2*border_size,240-2*border_size) # default values set for mini to get 3d

    print('roi = ', roi)
    print('press q on image to exit')

    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (f0.shape[1], f0.shape[0]), isColor=True)

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
