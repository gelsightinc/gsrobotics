#!usr/bin/python3.8

import cv2
import numpy as np
import time
import glob
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy import ndimage
from gelsight import gsdevice
from scipy.signal import fftconvolve


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
    :param image: N-D array    """
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


def find_marker(gray):
    adjusted = cv2.convertScaleAbs(gray, alpha=3, beta=0)
    mask = cv2.inRange(adjusted, 255, 255)
    ''' normalized cross correlation '''
    template = gkern(l=20, sig=5)

    nrmcrimg = normxcorr2(template, mask)
    ''''''''''''''''''''''''''''''''''''
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    a = nrmcrimg
    b = 2 * ((a - a.min()) / (a.max() - a.min())) - 1
    b = (b - b.min()) / (b.max()-b.min())
    mask = np.asarray(b<0.50)
    return (mask*255).astype('uint8')


def find2dpeaks(res):
    '''
    Create masks for all the dots. find 2D peaks.
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    '''
    img3 = res
    neighborhood_size = 20
    threshold = 1

    data_max = maximum_filter(img3, neighborhood_size)
    maxima = (img3 == data_max)
    data_min = minimum_filter(img3, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    xy = np.array(ndimage.center_of_mass(img3, labeled, range(1, num_objects + 1)))

    # plt.imshow(img3)
    # plt.autoscale(False)
    # plt.plot(xy[:, 1], xy[:, 0], 'ro', markersize=2)
    # plt.show()
    # plt.close()

    return xy



if __name__ == "__main__":

    imgw = 640
    imgh = 480
    SAVE_VIDEO_FLAG = False
    USE_R1 = True

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(100, 100),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 2),
        flags=0)

    if USE_R1:
        gs = gsdevice.Camera(gsdevice.Finger.R1, 0)
        WHILE_COND = 1
    else:
        cap = cv2.VideoCapture('marker.avi') # choose to read from video or camera
        WHILE_COND = cap.isOpened()

    if SAVE_VIDEO_FLAG:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('flow.mov',fourcc, 60.0, (imgw,imgh)) # The fps depends on CPU

    gs.connect()
    if USE_R1:
        f0 = gs.get_image((0,0,320,240))
        f0 = cv2.resize(f0, (imgw, imgh))
        f0 = f0[40:460, 50:620]
        f0gray = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    else:
        ret, f0 = cap.read()

    # find mask
    mask = find_marker(f0gray)
    # find marker centers
    mc0 = find2dpeaks(mask)
    mc0[:, [0, 1]] = mc0[:, [1, 0]]
    mc0copy = mc0

    count = 0

    try:
        while (WHILE_COND):

            t0 = time.time()

            if USE_R1:
                curr = gs.get_image((0,0,320,240))
                curr = cv2.resize(curr, (imgw, imgh))
                curr = curr[40:460, 50:620]
                currgray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                # find mask
                mask = find_marker(currgray)
                # find marker centers
                mc = find2dpeaks(mask)
                mc[:, [0, 1]] = mc[:, [1, 0]]
                for i in range(mc.shape[0]):
                    x0, y0 = int(mc[i][0]), int(mc[i][1])
                    cv2.circle(mask, (x0, y0), color=(0, 0, 0), radius=1, thickness=2)
            else:
                ret, curr = cap.read()

            if count==0:
                prev = f0
                # mc = mc0
                count += 1

            p2, st2, err2 = cv2.calcOpticalFlowPyrLK(f0, curr, mc0.astype('float32'), None, **lk_params)
            # p2, st2, err2 = cv2.calcOpticalFlowPyrLK(prev, curr, mc0copy.astype('float32'), mc.astype('float32'), **lk_params)

            # Select good points
            # good_p2 = p2[(st2==1).reshape(-1),:]

            for i in range(p2.shape[0]):
                x0,y0 = int( mc0[i][0]),int(mc0[i][1])
                x1,y1 = int(p2[i][0]), int(p2[i][1])
                cv2.arrowedLine(curr, (x0,y0), (x1,y1), (0, 255, 255), thickness=2, tipLength=0.25)
                # x0, y0 = int(mc0[i][0]), int(mc0[i][1])
                # cv2.circle(curr,(x0,y0), color=(0, 0, 255),radius=1,thickness=1)

            mc0copy = p2
            prev = curr.copy()

            cv2.imshow('frame', curr)
            cv2.imshow('mask', mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if SAVE_VIDEO_FLAG:
                out.write(curr)

        if USE_R1:
            gs.stop_video()
        else:
            cap.release()
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
            print('Interrupted!')
            gs.stop_video()

    gs.stop_video()
