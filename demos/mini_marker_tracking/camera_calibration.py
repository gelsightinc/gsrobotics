import cv2
import numpy as np
import glob
import pickle as pk
# import cv2.aruco as aruco
#import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
# from Robotics.tactile_sdk_local.marker_tracking.tracking.src import setting
# from Robotics.tactile_sdk_local.marker_tracking.tracking.src.marker_dectection import *
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy import ndimage
from scipy.signal import fftconvolve



'''
USEFUL LINKS:
1. http://ais.informatik.uni-freiburg.de/teaching/ws09/robotics2/pdfs/rob2-08-camera-calibration.pdf
2. https://stackoverflow.com/questions/55220229/extrinsic-matrix-computation-with-opencv
3. https://learnopencv.com/geometry-of-image-formation/
'''


def gkern(l=5, sig=1.):
    """ creates gaussian kernel with side length l and a sigma of sig """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

def normxcorr2(template, image, mode="same"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")
    template = template - np.mean(template)
    image = image - np.mean(image)
    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))
    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0
    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)
    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    return out


def find_marker(frame):
    ##### masking techinique for big dots
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # res = preprocessimg(gray)
    # mask = cv2.inRange(gray, 20, 70)
    # kernel = np.ones((3, 3), np.uint8)
    # dilation = cv2.dilate(mask, kernel, iterations=1)
    adjusted = cv2.convertScaleAbs(gray, alpha=3, beta=1)
    mask = cv2.inRange(adjusted, 150, 255)
    ''' normalized cross correlation '''
    template = gkern(l=10, sig=5)
    nrmcrimg = normxcorr2(template, mask)
    ''''''''''''''''''''''''''''''''''''
    a = nrmcrimg
    b = 2 * ((a - a.min()) / (a.max() - a.min())) - 1
    b = (b - b.min()) / (b.max() - b.min())
    mask = np.asarray(b < 0.50)
    return (mask * 255).astype('uint8')


def marker_center(mask, frame):
    ''' second method '''
    img3 = mask
    neighborhood_size = 10
    threshold = 40
    data_max = maximum_filter(img3, neighborhood_size)
    maxima = (img3 == data_max)
    data_min = minimum_filter(img3, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    MarkerCenter = np.array(ndimage.center_of_mass(img3, labeled, range(1, num_objects + 1)))
    MarkerCenter[:, [0, 1]] = MarkerCenter[:, [1, 0]]
    for i in range(MarkerCenter.shape[0]):
        x0, y0 = int(MarkerCenter[i][0]), int(MarkerCenter[i][1])
        cv2.circle(mask, (x0, y0), color=(0, 0, 0), radius=1, thickness=1)
    return MarkerCenter


def order_points(pts):
    ## order points in clockwise direction, starting from top left
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")


def calibrate_camera_intrinsic():
    '''
    Calibrate camera with a checkboard pattern to get intrinsic params
        LINK: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
    :return:
    '''
    NUM_SQ_ROW = 8
    NUM_SQ_COL = 10

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((NUM_SQ_ROW * NUM_SQ_COL, 3), np.float32)
    objp[:, :2] = np.mgrid[0:NUM_SQ_ROW, 0:NUM_SQ_COL].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    path = '/home/radhen/Documents/test_images/r2d2/cam_calib_imgs/mini1_rawimgs_trial1/'
    images = glob.glob(path + '/*.jpg')
    # '/home/radhen/Documents/test_images/googlex_calib_instrinsic_imgs/r15005'
    accept_cnt = 0
    reject_cnt = 0

    for fname in images:

        img = cv2.imread(fname)
        # img = img[110:540, 55:425]
        # img = img[100:550, 20:460]
        # img = warp_perspective(img, [[25, 25], [345, 25], [310, 380], [60, 380]], output_sz=(320, 240))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=2.5, beta=0)

        # Find the chess board corners
        flags = None
        # flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
        # flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, (NUM_SQ_ROW, NUM_SQ_COL), flags)

        # If found, add object points, image points (after refining them)
        if ret == True:
            accept_cnt += 1; print ('Accepted: ', accept_cnt)

            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (NUM_SQ_ROW, NUM_SQ_COL), corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

            # Save the calibrated image
            savepath = '/home/radhen/Documents/test_images/r2d2/cam_calib_imgs/calibrated_imgs_trial1/'
            cv2.imwrite(savepath + 'img_{}.jpg'.format(accept_cnt), img)
        else:
            reject_cnt += 1;
            print('Rejected: ', reject_cnt)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print ("mean error: ", mean_error / len(objpoints))

    ########## SAVE THE CAMERA PARAMS ###################
    with open(path + '/' + "objpoints.txt", "wb") as fp:  # Pickling
        pk.dump(objpoints, fp)

    with open(path + '/' + "imgpoints.txt", "wb") as fp:  # Pickling
        pk.dump(imgpoints, fp)

    with open(path + '/' + "ret.txt", "wb") as fp:  # Pickling
        pk.dump(ret, fp)

    with open(path + '/' + "mtx.txt", "wb") as fp:  # Pickling
        pk.dump(mtx, fp)

    with open(path + '/' + "dist.txt", "wb") as fp:  # Pickling
        pk.dump(dist, fp)

    with open(path + '/' + "rvecs.txt", "wb") as fp:  # Pickling
        pk.dump(rvecs, fp)

    with open(path + '/' + "tvecs.txt", "wb") as fp:  # Pickling
        pk.dump(tvecs, fp)


def calibrate_camera_intrinsic_v2(img):

    img = img[30:380,80:355]
    ### find marker masks
    marker_mask = find_marker(img)

    ### find marker centers
    mc = marker_center(marker_mask, img)

    print ('')



def undistort_img(image_name):
    '''
    :param image_name: pass the name of image (str)
    :return: undistorted image
    '''
    ########## LOAD THE CAMERA PARAMS ##################
    with open("mtx.txt", "rb") as fp:  # Unpickling
        mtx = pk.load(fp)

    with open("dist.txt", "rb") as fp:  # Unpickling
        dist = pk.load(fp)

    ########## Undistort the image ##################
    img = cv2.imread(image_name)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]
    # cv2.imwrite('calibresult.png', dst)

    return dst


def warp_perspective(img, corners, output_sz):
    TOPLEFT, TOPRIGHT, BOTTOMRIGHT, BOTTOMLEFT = corners
    WARP_H = output_sz[0]
    WARP_W = output_sz[1]
    points1=np.float32([TOPLEFT,TOPRIGHT,BOTTOMRIGHT, BOTTOMLEFT])
    points2=np.float32([[0,0],[WARP_W,0],[WARP_W,WARP_H], [0, WARP_H]])
    matrix=cv2.getPerspectiveTransform(points1,points2)
    result = cv2.warpPerspective(img, matrix, (WARP_W, WARP_H))
    return result


def save_params_to_file(corners):
    BOTTOM_LEFT = corners[4]
    TOP_LEFT = corners[39]
    BOTTOM_LEFT = corners[0]
    TOP_RIGHT = corners[35]

    file1 = open("camera_extrinsic_params_r1.5.txt", "w")
    L = ["BOTTOM_LEFT: ", np.array_str(BOTTOM_LEFT), "\n",
         "TOP_LEFT: ", np.array_str(TOP_LEFT), "\n",
         "BOTTOM_RIGHT: ", np.array_str(BOTTOM_RIGHT), "\n",
         "TOP_RIGHT: ", np.array_str(TOP_RIGHT), "\n"]
    # \n is placed to indicate EOL (End of Line)
    file1.write("\n\n Corners found using checkerboard pattern for R1.5 sensor \n\n")
    file1.writelines(L)
    file1.close()  # to change file access modes


def find_corners(img):
    ''' automatic wrap '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # compute a "wide", "mid-range", and "tight" threshold for the edges
    # using the Canny edge detector
    wide = cv2.Canny(blurred, 10, 200)
    mid = cv2.Canny(blurred, 30, 150)
    tight = cv2.Canny(blurred, 240, 250)

    # ### find masks
    mask = (gray<50)*1.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=5)

    dst = cv2.dilate(cv2.cornerHarris(mask.astype('float32'),15,5,0.5),None)
    corners = cv2.goodFeaturesToTrack(dst, 4, 0.25, 150)

    corners = corners.reshape(4,2)
    oc = order_points(corners)

    # plt.imshow(img)
    # plt.scatter(oc[:, 0], oc[:, 1])
    # plt.show()

    ### tidy up the coordinated
    if oc[1,1] <= oc[0,1]:
        oc[1,1] = oc[0,1]
    else:
        oc[0,1] = oc[1,1]
    if oc[2, 1] <= oc[3, 1]:
        oc[3, 1] = oc[2, 1]
    else:
        oc[3, 1] = oc[2, 1]
    if (oc[3,0]-oc[0,0]) >= (oc[1,0]-oc[2,0]):
        oc[2,0] = oc[2,0] - ((oc[3,0]-oc[0,0])-(oc[1,0]-oc[2,0]))
    else:
        oc[3,0] = oc[3,0] + ((oc[1,0]-oc[2,0])-(oc[3, 0]-oc[0,0]))

    boundrary = 12
    oc[0,:] = oc[0,:] + boundrary
    oc[1,0] = oc[1,0] - boundrary
    oc[1,1] = oc[1,1] + boundrary
    oc[2,:] = oc[2,:] - boundrary
    oc[3,0] = oc[3,0] + boundrary
    oc[3,1] = oc[3,1] - boundrary

    res = warp_perspective(img, oc, output_sz=(320,240))

    ### find marker masks
    marker_mask = find_marker(res)

    ### find marker centers
    mc = marker_center(marker_mask, res)

    return res, oc



if __name__ == "__main__":

    '''
    Fetch the calibration image from live camera
    '''
    # img = get_image('http://raspiatgelsightinc.local:8080/?action=stream')
    # plt.imsave('/home/radhen/Documents/tactile_sdk/scripts/utils/camera_calib_imgs/extrinsic_calib_img.png', img)
    # img = img[110:540, 55:425]

    '''
    Load image
    '''
    # img = cv2.imread('/home/radhen/Documents/test_images/tricam/stereo_calib_imgs/40px/cam0/cam0-2.png')
    # img = img[100:550,20:460] # chop off the black and mirrored region
    # img_gray = cv2.imread('/home/radhen/Documents/tactile_sdk/scripts/utils/camera_calib_imgs/intrinsic_imgs/2700_single_img.png', 0)
    # img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)
    # img_gray = cv2.flip(img_gray, -1)
    # img_gray = img_gray[110:540, 55:425]



    '''
    Camera INTRINSIC calibration
    '''
    ### find calib params
    calibrate_camera_intrinsic()
    # calibrate_camera_intrinsic_v2(img)

    #### USE PARAMS ALREADY FOUND
    ### TRICAM0
    # camera_matrix = np.asarray([[199, 0., 202],      # [[f_x, 0, c_x],
    #                             [0., 199, 193],        # [0, f_y, c_y],
    #                             [0., 0.,  1.]])     # [0,  0,  1]]
    #
    # distortion_coeff = np.asarray([[0.1247, -0.1724, 0.0001, -0.0005, 0.0428]])   ## [k_1, k_2, p_1, p_2, k_3]

    ########## LOAD THE CAMERA PARAMS ##################
    # path = '/home/radhen/Documents/test_images/r2d2/cam_calib_imgs/raw_imgs/'
    # with open(path + "mtx.txt", "rb") as fp:  # Unpickling
    #     mtx = pk.load(fp)

    # with open("dist.txt", "rb") as fp:  # Unpickling
    #     dist = pk.load(fp)

    # h, w = img.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w, h), 1, (w, h))
    # # undistort
    # dst = cv2.undistort(img, camera_matrix, distortion_coeff, None, newcameramtx)


    '''
    Find out camera EXTRINSIC params using Aruco markers
    '''
    # img = cv2.imread('/home/radhen/Documents/tactile_sdk/scripts/utils/extrinsic_calib_test_1.png')
    # img_cropped = img[90:390, 170:450]
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    # parameters = aruco.DetectorParameters_create()
    # corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)


    '''
    Find out camera EXTRINSIC params using checker board
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
    '''
    # BOTTOM_LEFT, TOP_LEFT, BOTTOM_RIGHT, TOP_RIGHT = [0] * 4
    # ### termination criteria
    # NUM_SQ_ROW = 8
    # NUM_SQ_COL = 5
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # objp = np.zeros((NUM_SQ_COL * NUM_SQ_ROW, 3), np.float32)
    # objp[:, :2] = np.mgrid[0:NUM_SQ_ROW, 0:NUM_SQ_COL].T.reshape(-1, 2)
    # # Arrays to store object points and image points from all the images.
    # objpoints = []  # 3d point in real world space
    # imgpoints = []  # 2d points in image plane.
    #
    # ### Find the chess board corners
    # flags = 0
    # flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
    # flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
    # ret, corners = cv2.findChessboardCorners(img_gray, (NUM_SQ_ROW, NUM_SQ_COL), flags)
    #
    # if ret == True:
    #     objpoints.append(objp)
    #
    #     corners2 = cv2.cornerSubPix(img_gray, corners, (5, 5), (-1, -1), criteria)
    #     imgpoints.append(corners2)
    #
    #     # Draw and display the corners
    #     img = cv2.drawChessboardCorners(img_gray, (NUM_SQ_ROW, NUM_SQ_COL), corners2, ret)
    #     # plt.imshow(img)
    #     plt.imsave('/home/radhen/Documents/tactile_sdk/scripts/utils/camera_calib_imgs/calib_marker_detection_8.png', img)
    # #
    # # save_params_to_file(corners)


    '''
    Unwrap perspective. Manual locations (EXTRINSIC CALIBRATION)
    '''

    ### camera params for gel with biggest black dots
    # img = warp_perspective(img_gray, [[30, 20], [360, 20], [60, 360], [316, 360]], output_sz=img_gray.shape)

    #### camera params for gel number #51
    # img = warp_perspective(img_gray, [[69, 18], [262, 19], [81, 345], [255, 342]], output_sz=img.shape)
    # plt.imsave('/home/radhen/Documents/tactile_sdk/scripts/utils/camera_calib_imgs/calibrated_img_.png', img)
    # plt.imshow(img)
    # print ("done")


    '''
    Unwrap perspective. Automatic corners (EXTRINSIC CALIBRATION)
    '''
    # img, oc = find_corners(img)