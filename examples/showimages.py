import sys
import time
import cv2
import gsdevice

def main(argv):

    # Set flags 
    SAVE_VIDEO_FLAG = False
    USE_ROI = False

    # the device ID can change after unplugging and changing the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    dev = gsdevice.Camera("GelSight Mini")

    dev.connect()

    f0 = dev.get_raw_image()
    #print('image size = ', f0.shape[1], f0.shape[0])
    roi = (0, 0, f0.shape[1], f0.shape[0])

    if USE_ROI:
        print('Select an ROI in the ROI window\n')
        roi = cv2.selectROI(f0)
        roi_cropped = f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv2.imshow('ROI', roi_cropped)
        print('Press q in ROI image to continue\n')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('roi = ', roi)

    print('press q on image to exit')

    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (f0.shape[1], f0.shape[0]), isColor=True)
        print(f'Saving data to {file_path}')
    try:
        while dev.while_condition:

            # get the roi image
            f1 = dev.get_image()
            if USE_ROI:
                f1 = f1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            bigframe = cv2.resize(f1, (f1.shape[1]*2, f1.shape[0]*2))
            cv2.imshow('Image', bigframe)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if SAVE_VIDEO_FLAG:
                out.write(f1)

    except KeyboardInterrupt:
            print('Interrupted!')
            dev.stop_video()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])
