import copy
import find_marker
import numpy as np
import cv2
import time
import marker_detection
import sys
import setting
import os

def find_cameras():
    # checks the first 10 indexes.
    index = 0
    arr = []
    if os.name == 'nt':
        cameras = find_cameras_windows()
        return cameras
    i = 10
    while i >= 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            command = 'v4l2-ctl -d ' + str(index) + ' --info'
            is_arducam = os.popen(command).read()
            if is_arducam.find('Arducam') != -1 or is_arducam.find('Mini') != -1:
                arr.append(index)
            cap.release()
        index += 1
        i -= 1

    return arr

def find_cameras_windows():
    # checks the first 10 indexes.
    index = 0
    arr = []
    idVendor = 0xC45
    idProduct = 0x636D
    import usb.core
    import usb.backend.libusb1
    backend = usb.backend.libusb1.get_backend(
        find_library=lambda x: "libusb_win/libusb-1.0.dll"
    )
    dev = usb.core.find(backend=backend, find_all=True)
    # loop through devices, printing vendor and product ids in decimal and hex
    for cfg in dev:
        #print('Decimal VendorID=' + hex(cfg.idVendor) + ' & ProductID=' + hex(cfg.idProduct) + '\n')
        if cfg.idVendor == idVendor and cfg.idProduct == idProduct:
            arr.append(index)
        index += 1

    return arr


def resize_crop_mini(img, imgw, imgh):
    # resize, crop and resize back
    img = cv2.resize(img, (895, 672))  # size suggested by janos to maintain aspect ratio
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
    img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    img = img[:, :-1]  # remove last column to get a popular image resolution
    img = cv2.resize(img, (imgw, imgh))  # final resize for 3d
    return img

def trim(img):
    img[img<0] = 0
    img[img>255] = 255



def compute_tracker_gel_stats(thresh):
    numcircles = 9 * 7;
    mmpp = .063;
    true_radius_mm = .5;
    true_radius_pixels = true_radius_mm / mmpp;
    circles = np.where(thresh)[0].shape[0]
    circlearea = circles / numcircles;
    radius = np.sqrt(circlearea / np.pi);
    radius_in_mm = radius * mmpp;
    percent_coverage = circlearea / (np.pi * (true_radius_pixels) ** 2);
    return radius_in_mm, percent_coverage*100.


def main(argv):

    imgw = 320
    imgh = 240

    calibrate = False

    outdir = './TEST/'
    SAVE_VIDEO_FLAG = False
    SAVE_ONE_IMG_FLAG = False
    SAVE_DATA_FLAG = False

    if SAVE_ONE_IMG_FLAG:
        sn = input('Please enter the serial number of the gel \n')
        #sn = str(5)
        viddir = outdir + 'vids/'
        imgdir = outdir + 'imgs/'
        resultsfile = outdir + 'marker_qc_results.txt'
        vidfile = viddir + sn + '.avi'
        imgonlyfile = imgdir + sn + '.png'
        maskfile = imgdir + 'mask_' + sn + '.png'
        # check to see if the directory exists, if not create it
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if not os.path.exists(viddir):
            os.mkdir(viddir)
        if not os.path.exists(imgdir):
            os.mkdir(imgdir)

    if SAVE_DATA_FLAG:
        datadir = outdir + 'data'
        datafilename = datadir + 'marker_locations.txt'
        datafile = open(datafilename,"a")

    # if len(sys.argv) > 1:
    #     if sys.argv[1] == 'calibrate':
    #         calibrate = True

    cameras = find_cameras()
    cap = cv2.VideoCapture(cameras[0])
    WHILE_COND = cap.isOpened()

    # set the format into MJPG in the FourCC format
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))

    # Resize scale for faster image processing
    setting.init()

    if SAVE_VIDEO_FLAG:
        # Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fourcc = cv2.VideoWriter_fourcc('p', 'n', 'g', ' ')
        # fourcc = cv2.VideoWriter_fourcc(*'jp2','')
        out = cv2.VideoWriter(vidfile, fourcc, 25, (imgw, imgh), isColor=True)

    frame0 = None

    counter = 0
    while 1:
        if counter<50:
            ret, frame = cap.read()
            print ('flush black imgs')

            if counter == 48:
                ret, frame = cap.read()
                ##########################
                frame = resize_crop_mini(frame, imgw, imgh)
                ### find marker masks
                mask = marker_detection.find_marker(frame)
                ### find marker centers
                mc = marker_detection.marker_center(mask, frame)
                break

            counter += 1

    counter = 0

    mccopy = mc
    mc_sorted1 = mc[mc[:,0].argsort()]
    mc1 = mc_sorted1[:setting.N_]
    mc1 = mc1[mc1[:,1].argsort()]

    mc_sorted2 = mc[mc[:,1].argsort()]
    mc2 = mc_sorted2[:setting.M_]
    mc2 = mc2[mc2[:,0].argsort()]


    """
    N_, M_: the row and column of the marker array
    x0_, y0_: the coordinate of upper-left marker
    dx_, dy_: the horizontal and vertical interval between adjacent markers
    """
    N_= setting.N_
    M_= setting.M_
    fps_ = setting.fps_
    x0_ = np.round(mc1[0][0])
    y0_ = np.round(mc1[0][1])
    dx_ = mc2[1, 0] - mc2[0, 0]
    dy_ = mc1[1, 1] - mc1[0, 1]

    print ('x0:',x0_,'\n', 'y0:', y0_,'\n', 'dx:',dx_,'\n', 'dy:', dy_)

    radius ,coverage = compute_tracker_gel_stats(mask)

    if SAVE_ONE_IMG_FLAG:
        fresults = open(resultsfile, "a")
        fresults.write(f"{sn} {float(f'{dx_:.2f}')} {float(f'{dy_:.2f}')} {float(f'{radius*2:.2f}')} {float(f'{coverage:.2f}')}\n")


    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('frame', 320*3, 240*3)
    # cv2.resizeWindow('mask', 320*3, 240*3)
    # Create Mathing Class

    m = find_marker.Matching(N_,M_,fps_,x0_,y0_,dx_,dy_)

    frameno = 0
    try:
        while (WHILE_COND):

            ret, frame = cap.read()
            if not(ret):
                break

            ##########################
            # resize (or unwarp)
            # frame = cv2.resize(frame, (imgw,imgh))
            # frame = frame[10:-10,5:-10] # for R1.5
            # frame = frame[border_size:imgh - border_size, border_size:imgw - border_size] # for mini
            frame = resize_crop_mini(frame, imgw, imgh)
            raw_img = copy.deepcopy(frame)

            ''' EXTRINSIC calibration ... 
            ... the order of points [x_i,y_i] | i=[1,2,3,4], are same 
            as they appear in plt.imshow() image window. Put them in 
            clockwise order starting from the topleft corner'''
            # frame = warp_perspective(frame, [[35, 15], [320, 15], [290, 360], [65, 360]], output_sz=frame.shape[:2])   # params for small dots
            # frame = warp_perspective(frame, [[180, 130], [880, 130], [800, 900], [260, 900]], output_sz=(640,480)) # org. img size (1080x1080)

            ### find marker masks
            mask = marker_detection.find_marker(frame)

            ### find marker centers
            mc = marker_detection.marker_center(mask, frame)

            if calibrate == False:
                tm = time.time()
                ### matching init
                m.init(mc)

                ### matching
                m.run()
                # print(time.time() - tm)

                ### matching result
                """
                output: (Ox, Oy, Cx, Cy, Occupied) = flow
                    Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
                    Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
                    Occupied: N*M matrix, the index of the marker at each position, -1 means inferred. 
                        e.g. Occupied[i][j] = k, meaning the marker mc[k] lies in row i, column j.
                """
                flow = m.get_flow()

                if frame0 is None:
                    frame0 = frame.copy()
                    frame0 = cv2.GaussianBlur(frame0, (int(63), int(63)), 0)

                # diff = (frame * 1.0 - frame0) * 4 + 127
                # trim(diff)

                # # draw flow
                marker_detection.draw_flow(frame, flow)

                frameno = frameno + 1

                if SAVE_DATA_FLAG:
                    Ox, Oy, Cx, Cy, Occupied = flow
                    for i in range(len(Ox)):
                        for j in range(len(Ox[i])):
                            datafile.write(
                               f"{frameno}, {i}, {j}, {Ox[i][j]:.2f}, {Oy[i][j]:.2f}, {Cx[i][j]:.2f}, {Cy[i][j]:.2f}\n")

            #mask_img = mask.astype(frame[0].dtype)
            mask_img = np.asarray(mask)
            # mask_img = cv2.merge((mask_img, mask_img, mask_img))

            bigframe = cv2.resize(frame, (frame.shape[1]*3, frame.shape[0]*3))
            cv2.imshow('frame', bigframe)
            bigmask = cv2.resize(mask_img*255, (mask_img.shape[1]*3, mask_img.shape[0]*3))
            cv2.imshow('mask', bigmask)

            if SAVE_ONE_IMG_FLAG:
                cv2.imwrite(imgonlyfile, raw_img)
                cv2.imwrite(maskfile, mask*255)
                SAVE_ONE_IMG_FLAG = False

            if calibrate:
                ### Display the mask
                cv2.imshow('mask',mask_img*255)
            if SAVE_VIDEO_FLAG:
                out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print('Interrupted!')

    ### release the capture and other stuff
    cap.release()
    cv2.destroyAllWindows()
    if SAVE_VIDEO_FLAG:
        out.release()

if __name__ == "__main__":
    main(sys.argv[1:])
