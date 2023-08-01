import os
import time
import numpy as np
import cv2
import torch
from gs_utils import demark
from gs_utils import RGB2NormNet
from gs_utils import poisson_dct_neumaan
from gs_utils import Visualize3D
from gs_utils import write_tmd
from gelsight import gsdevice
from markertracker import MarkerTracker

def compute_depth_map(img, markermask, net, device):

    ''' Set the contact mask to everything but the markers '''
    cm = ~markermask
    #m.init(mc)

    '''intersection of cm and markermask '''
    ''' find contact region '''
    #cm, cmindx = np.ones(img.shape[:2]), np.where(np.ones(img.shape[:2]))
    # cmandmm = (np.logical_and(cm, markermask)).astype('uint8')

    ''' Get depth image with NN '''
    nx = np.zeros(img.shape[:2])
    ny = np.zeros(img.shape[:2])
    dm = np.zeros(img.shape[:2])

    ''' ENTIRE CONTACT MASK THRU NN '''
    # if np.where(cm)[0].shape[0] != 0:
    rgb = img[np.where(cm)] / 255
    pxpos = np.vstack(np.where(cm)).T
    pxpos[:, 0], pxpos[:, 1] = pxpos[:, 0] / img.shape[0], pxpos[:, 1] / img.shape[1]
    features = np.column_stack((rgb, pxpos))
    features = torch.from_numpy(features).float().to(device)

    with torch.no_grad():
        net.eval()
        out = net(features)

    nx[np.where(cm)] = out[:, 0].cpu().detach().numpy()
    ny[np.where(cm)] = out[:, 1].cpu().detach().numpy()

    '''OPTION#2 calculate gx, gy from nx, ny. '''
    nz = np.sqrt(1 - nx ** 2 - ny ** 2)
    if np.isnan(nz).any():
        print('nan found')

    nz = np.nan_to_num(nz)
    gx = -np.divide(nx, nz)
    gy = -np.divide(ny, nz)

 #   dilated_mm = dilate(markermask, ksize=3, iter=2)
    gx_interp, gy_interp = demark(gx, gy, markermask)

    dm = poisson_dct_neumaan(gx_interp, gy_interp)

    return dm

def process_marker_gel(video_filename, nnet_filename, options, output_filepath=''):

    # Options
    USE_MINI_LIVE = options['USE_MINI_LIVE']
    GPU = options['GPU']
    SHOW_3D = options['SHOW_3D']
    SAVE_MARKER_FLAG = options['SAVE_MARKER_FLAG']
    SAVE_VIDEO_FLAG = options['SAVE_VIDEO_FLAG']
    SAVE_3D_TMD = options['SAVE_3D_TMD']
    SAVE_3D_PCD = options['SAVE_3D_PCD']

    # Local option to draw markers since they don't get correctly recorded if the video is saved
    DRAW_MARKERS = True
    if SAVE_VIDEO_FLAG:
        DRAW_MARKERS = False

    marker_savepath = output_filepath

    cf = np.zeros(3)

    # Image dimensions
    imgw = 320
    imgh = 240

    mmpp = .063  ## for Mini at 320x240
    mpp = mmpp / 1000

    counter = 0

    # This is for saving individual frames of data
    tmdcounter = 0
    pcdcounter = 0

    if SHOW_3D:
        # Check that input files exist
        if not os.path.isfile(nnet_file):
            print(f'Input model file does not exits {nnet_file}')
            exit(-1)

        ''' Load neural network '''
        if GPU:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        net = RGB2NormNet().float().to(device)
        if GPU:
            ### load weights on gpu
            # net.load_state_dict(torch.load(net_path))
            checkpoint = torch.load(nnet_filename, map_location=lambda storage, loc: storage.cuda(0))
            net.load_state_dict(checkpoint['state_dict'])
        else:
            ### load weights on cpu which were actually trained on gpu
            checkpoint = torch.load(nnet_filename, map_location=lambda storage, loc: storage)
            net.load_state_dict(checkpoint['state_dict'])

        ''' use this to plot just the 3d '''
        vis3d = Visualize3D(imgw, imgh, mmpp, '')

    needcount = True;
    if SAVE_MARKER_FLAG:
        if not os.path.exists(marker_savepath):
            os.mkdir(marker_savepath)
        count = 0
        while needcount:
            outfilename = 'marker_locations_' + str(count) + '.csv'
            outfile = os.path.join(marker_savepath, outfilename)
            if os.path.isfile(outfile):
                count = count + 1;
            else:
                needcount = False;
        datafile = open(outfile, "w")
        frameno = 0
        print(f'Saving data to {outfile}')

    if SAVE_VIDEO_FLAG:
        if needcount:
            while needcount:
                vidfilename = 'video_' + str(count) + '.avi'
                vidfile = os.path.join(marker_savepath, vidfilename)
                if os.path.isfile(vidfile):
                    count = count + 1;
                else:
                    needcount = False;
        else:
            vidfilename = 'video_' + str(count) + '.avi'
            vidfile = os.path.join(marker_savepath, vidfilename)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        markervid = cv2.VideoWriter(vidfile, fourcc, 30, (imgw, imgh), isColor=True)
        print(f'Saving data to video to {vidfile}')

    # Get mask
    WARMUP = 10
    frame0 = None

    if USE_MINI_LIVE:
        gs = gsdevice.Camera("GelSight Mini")
        WHILE_COND = 1
        gs.connect()
    else:
        vidinput = cv2.VideoCapture(video_filename)
        WHILE_COND = vidinput.isOpened()
        if not vidinput.isOpened():
            print(f'Error opening video {video_filename}')

    print('--- Getting initial markers. Gel should not be touching anything ---')
    counter = 0
    NUM_INIT_FRAMES = 10
    while counter < NUM_INIT_FRAMES:
        if USE_MINI_LIVE:
            frame0 = gs.get_raw_image()
        else:
            ret, frame0 = vidinput.read()

        frame = cv2.resize(frame0, (imgw, imgh))
        counter = counter + 1
        if counter == NUM_INIT_FRAMES:
            img = np.float32(frame) / 255.0
            mtracker = MarkerTracker(img)

    marker_centers = mtracker.marker_center
    Ox = marker_centers[:, 1]
    Oy = marker_centers[:, 0]
    nct = len(marker_centers)

    if SHOW_3D:
        # Retrieve and display images
        NUM_INIT_FRAMES = 30
        dm_zero_counter = 0
        dm_zero = np.zeros(frame.shape[:2])
        dm = np.zeros(frame.shape[:2])
        Oz = np.zeros(frame.shape[:2])

        print('--- Processing zeroing depth. Gel should not be touching anything ---')
        while (dm_zero_counter < NUM_INIT_FRAMES):

            if USE_MINI_LIVE:
                image_in = gs.get_raw_image()
            else:
                ret, image_in = vidinput.read()
                if not ret:
                    print(f'Error reading video {video_filename}')

            img = cv2.resize(image_in, (imgw, imgh))

            markermask = mtracker.marker_mask
            markermask = markermask.astype('uint8')
            dm = compute_depth_map(img, markermask, net, device)
            # put the depth map in mm
            # dm = dm * mmpp * 1

            ''' remove initial zero depth '''
            if dm_zero_counter < NUM_INIT_FRAMES:
                dm_zero = dm_zero + dm
                if dm_zero_counter == NUM_INIT_FRAMES - 1:
                    dm_zero = dm_zero / dm_zero_counter
                    print(f'Finished processing zero depth.')
            dm_zero_counter += 1
            dm = dm - dm_zero

        Oz = dm

    print('press q on display to quit')
    # Retrieve and display images
    frame_count = 0
    while (WHILE_COND):

        if USE_MINI_LIVE:
            frame = gs.get_image()
            if frame.shape[1] == imgw and frame.shape[0] == imgh:
                success = True
        else:
            success, frame = vidinput.read()

        if success != True:
            cv2.destroyAllWindows()
            break

        if success:
            image_in = cv2.resize(frame, (imgw, imgh))

            img = np.float32(image_in) / 255.0
            mtracker.track_markers(img)

            currentpos = mtracker.marker_currentpos

            # Find the markers
            markermask = mtracker.marker_mask
            markermask = markermask.astype('uint8')

            # Compute the depth
            if SHOW_3D:
                dm = compute_depth_map(image_in, markermask, net, device)
                # put the depth map in mm
                # dm = dm * mmpp * 1
                dm = dm - dm_zero

            centers = marker_centers
            pts = currentpos
            for c in range(nct):
                pt1 = (int(centers[c, 1]), int(centers[c, 0]))
                pt2 = (int(pts[c, 1]), int(pts[c, 0]))
                cv2.arrowedLine(image_in, pt1, pt2, (255,255,255), thickness=1, line_type=cv2.LINE_8, tipLength=.15)
                #cv2.circle(image_in, pt1, int(mtracker.marker_radius[c]/2.), (0, 255, 255), 1);
                if DRAW_MARKERS:
                    cv2.drawMarker(image_in, pt1, markerType=cv2.MARKER_STAR, color=(0, 0, 255), markerSize=2)
                    cv2.drawMarker(image_in, pt2, markerType=cv2.MARKER_CROSS, color=(0, 255, 0), markerSize=2)
            cv2.imshow('Marker Frame', cv2.resize(image_in, (2 * img.shape[1], 2 * img.shape[0])))
            #cv2.imshow('Marker Mask', cv2.resize(markermask, (2*markermask.shape[1], 2*markermask.shape[0])))
            cv2.waitKey(10)

            ''' visualize 3d '''
            if SHOW_3D:
                vis3d.update(dm*mmpp)

            ### find marker masks
            mask = mtracker.marker_mask

            ### find marker centers
            mc = mtracker.marker_center

            ### matching result
            """
            Ox, Oy, Cx, Cy
                Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
                Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
            """

            # draw flow

            # write the marker frame to the video and marker locations to a file
            if SAVE_VIDEO_FLAG:
                markervid.write(image_in)  # cv2.resize(img, (imgw, imgh)))

            frame_count = frame_count + 1
            if SAVE_MARKER_FLAG:

                Cx = currentpos[:,1]
                Cy = currentpos[:,0]
                coords = mtracker.initial_marker_coord
                xcoords = coords[1]
                ycoords = coords[0]
                for i in range(nct):
                    row = int(Cy[i])
                    col = int(Cx[i])
                    mrow = int(ycoords[i])
                    mcol = int(xcoords[i])
                    if SHOW_3D:
                        datafile.write(
                                f"{frame_count:6d}, {mrow:3d}, {mcol:3d}, "
                                f"{Ox[i]:6.2f}, {Oy[i]:6.2f}, {Oz[row][col]:6.2f}, "
                                f"{Cx[i]:6.2f}, {Cy[i]:6.2f}, {dm[row][col]:6.2f}\n")
                    else:
                        datafile.write(
                            f"{frame_count:6d}, {mrow:3d}, {mcol:3d}, "
                            f"{Ox[i]:6.2f}, {Oy[i]:6.2f}, "
                            f"{Cx[i]:6.2f}, {Cy[i]:6.2f}\n")

            if SHOW_3D and SAVE_3D_TMD:
                tmdname = 'depthmap_{}.tmd'.format(frame_count)
                write_tmd(os.path.join(output_filepath, tmdname), dm, mmpp)
                print('saved ', tmdname, ' to ', output_filepath)
            if SHOW_3D and SAVE_3D_PCD:
                pcdname = 'pointcloud_{}.pcd'.format(frame_count)
                vis3d.save(os.path.join(output_filepath, pcdname))
                print('saved ', pcdname, ' to ', output_filepath)

            ''' Display the results '''
            mask_img = mask*255
            mask_img = mask_img.astype('uint8')
            #cv2.imshow('frame_rgb', cv2.resize(img, (2*imgw,2*imgh)))
            #cv2.imshow('mask', cv2.resize(mask_img, (2*imgw,2*imgh)))

            counter += 1
            if counter%250 == 0:
                print(f'Processed {counter} frames')

            keyval = cv2.waitKey(1) & 0xFF
            if keyval == ord('q'):
                print ('Exiting')
                break
            elif keyval == ord('t'):
                tmdname = 'depthmap{}.tmd'.format(tmdcounter)
                write_tmd(os.path.join(output_filepath, tmdname), dm, mmpp)
                print('saved ', tmdname, ' to ', output_filepath)
                tmdcounter += 1
            elif keyval == ord('p'):
                pcdname = 'pointcloud{}.pcd'.format(pcdcounter)
                vis3d.save(os.path.join(output_filepath, pcdname))
                print('saved ', pcdname, ' to ', output_filepath)
                pcdcounter += 1

    if USE_MINI_LIVE:
        gs.stop_video()
    else:
        vidinput.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    net_filename = "nnmini.pt"
    vid_filename = "mini_example.avi"

    user_dir = os.path.expanduser('~')
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(curr_dir,"data")

    # The nnet file is in the current directory
    model_file_path = os.path.join(curr_dir, "../../examples")
    nnet_file = os.path.join(model_file_path, net_filename)

    video_filename = os.path.join(data_dir, vid_filename)
    output_filepath = os.path.join(curr_dir, "MARKERS")

    options = {
               'USE_MINI_LIVE': False,
               'GPU': False,
               'SHOW_3D': False,
               'SAVE_MARKER_FLAG': True,
               'SAVE_VIDEO_FLAG': True,
               'SAVE_3D_TMD': False,
               'SAVE_3D_PCD': False
               }

    process_marker_gel(video_filename, nnet_file, options, output_filepath)

    print('Done processing video\n')
