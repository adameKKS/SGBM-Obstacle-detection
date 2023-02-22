import math
import cv2
import numpy as np
import os
from natsort import natsorted
import glob
import yaml
from yaml import SafeLoader

curr_dir = os.path.dirname(os.path.abspath(__file__))

ExTrackbars: bool = False  # True for Extended Trackbars
depth_threshold = 0.7
focal_length_cam0 = math.sqrt(458.654 ** 2 + 457.296 ** 2)
switch = 0


def fetch_calculate():
    with open(curr_dir + '\\cam0\\sensor.yaml') as f:
        data_cam0 = yaml.load(f, Loader=SafeLoader)
    with open(curr_dir + '\\cam1\\sensor.yaml') as f:
        data_cam1 = yaml.load(f, Loader=SafeLoader)

    Tb_c0 = np.array(data_cam0['T_BS']['data']).reshape(4, 4)  # cam0 to base transformation array
    kcam_L = np.zeros((3, 3))  # cam0 intrinsic matrix
    kcam_L[0][0] = np.array(data_cam0['intrinsics'])[0]
    kcam_L[1][1] = np.array(data_cam0['intrinsics'])[1]
    kcam_L[0][2] = np.array(data_cam0['intrinsics'])[2]
    kcam_L[1][2] = np.array(data_cam0['intrinsics'])[3]
    kcam_L[2][2] = 1.
    pcam_L = np.array(data_cam0['distortion_coefficients'])  # cam0 distortion_coefficients

    Tb_c1 = np.array(data_cam1['T_BS']['data']).reshape(4, 4)  # cam1 to base transformation array
    kcam_R = np.zeros((3, 3))  # cam1 intrinsic matrix
    kcam_R[0][0] = np.array(data_cam1['intrinsics'])[0]
    kcam_R[1][1] = np.array(data_cam1['intrinsics'])[1]
    kcam_R[0][2] = np.array(data_cam1['intrinsics'])[2]
    kcam_R[1][2] = np.array(data_cam1['intrinsics'])[3]
    kcam_R[2][2] = 1.
    pcam_R = np.array(data_cam1['distortion_coefficients'])  # cam1 distortion_coefficients

    T_LR = np.linalg.inv(Tb_c0) @ Tb_c1  # calculation of Tc0_c1 => cam1 -> cam0 transformation matrix
    print(T_LR)
    return T_LR, kcam_L, pcam_L, kcam_R, pcam_R


def nothing(x):
    pass


def rectify(T_LR, kcam_L, pcam_L, kcam_R, pcam_R):
    R = T_LR[:3, :3]
    T = T_LR[:3, -1]

    CH = cv2.stereoRectify(kcam_L, pcam_L, kcam_R, pcam_R, (752, 480), R, T, flags=cv2.CALIB_ZERO_DISPARITY)
    R1 = CH[0]
    R2 = CH[1]
    P1 = CH[2]
    P2 = CH[3]
    Q = CH[4]

    map1L, map2L = cv2.initUndistortRectifyMap(kcam_L, pcam_L, R1, P1, (752, 480),
                                               cv2.CV_32FC1)

    map1R, map2R = cv2.initUndistortRectifyMap(kcam_R, pcam_R, R2, P2, (752, 480),
                                               cv2.CV_32FC1)
    return map1L, map2L, map1R, map2R, Q


def load_images_from_folder(folder):
    img_list = natsorted(folder, key=lambda y: y.lower())
    return img_list


def convert_disp(disparity):
    disparityF = disparity.astype(float)
    maxv = np.max(disparityF.flatten())
    minv = np.min(disparityF.flatten())
    disparityF = 255.0 * (disparityF - minv) / (maxv - minv)
    disparityU = disparityF.astype(np.uint8)
    return disparityU


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth = depth_map[y, x]
        print("Depth at ({}, {}) is {}".format(x, y, depth))


if ExTrackbars:
    cv2.namedWindow('Parameters Tuner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Parameters Tuner', 1300, 400)
    cv2.createTrackbar('minDisparity', 'Parameters Tuner', 27, 50, nothing)
    cv2.createTrackbar('numDisparities', 'Parameters Tuner', 2, 30, nothing)
    cv2.createTrackbar('blockSize', 'Parameters Tuner', 3, 50, nothing)
    cv2.createTrackbar('uniquenessRatio', 'Parameters Tuner', 7, 50, nothing)
    cv2.createTrackbar('speckleWindowSize', 'Parameters Tuner', 3, 25, nothing)
    cv2.createTrackbar('speckleRange', 'Parameters Tuner', 32, 100, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'Parameters Tuner', 12, 25, nothing)
    cv2.createTrackbar('P1', 'Parameters Tuner', 8, 12, nothing)
    cv2.createTrackbar('P2', 'Parameters Tuner', 32, 40, nothing)
    cv2.createTrackbar('preFilterCap', 'Parameters Tuner', 63, 100, nothing)
    cv2.createTrackbar('vis_mult', 'Parameters Tuner', 1, 20, nothing)
    cv2.createTrackbar('WLS: lambda', 'Parameters Tuner', 10000, 20000, nothing)
    cv2.createTrackbar('WLS: sigma', 'Parameters Tuner', 5, 20, nothing)
    left_matcher = cv2.StereoSGBM_create()
else:
    cv2.namedWindow('Parameters Tuner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Parameters Tuner', 600, 600)
    cv2.createTrackbar('numDisparities', 'Parameters Tuner', 2, 16, nothing)
    cv2.createTrackbar('blockSize', 'Parameters Tuner', 3, 50, nothing)
    cv2.createTrackbar('WLS: lambda', 'Parameters Tuner', 10000, 20000, nothing)
    cv2.createTrackbar('WLS: sigma', 'Parameters Tuner', 5, 20, nothing)
    cv2.createTrackbar('vis_mult', 'Parameters Tuner', 1, 20, nothing)

    left_matcher = cv2.StereoSGBM_create()

gt_disparity_maps = os.path.join(curr_dir, 'cam1\\data')  # CAMERA 1 (RIGHT)
gt_disparity_maps2 = os.path.join(curr_dir, 'cam0\\data')  # CAMERA 0 (LEFT)
gt_disparity_map = load_images_from_folder(glob.glob(os.path.join(gt_disparity_maps, '*.png')))
gt_disparity_map2 = load_images_from_folder(glob.glob(os.path.join(gt_disparity_maps2, '*.png')))

i = 0
T_LR, kcam_L, pcam_L, kcam_R, pcam_R = fetch_calculate()
baseline = np.linalg.norm(T_LR[:3, -1])

map1L, map2L, map1R, map2R, Q = rectify(T_LR, kcam_L, pcam_L, kcam_R, pcam_R)

while True:
    imL = cv2.imread(gt_disparity_map2[i])
    imR = cv2.imread(gt_disparity_map[i])

    imRrecti = cv2.remap(imR, map1R, map2R, cv2.INTER_LINEAR)
    imLrecti = cv2.remap(imL, map1L, map2L, cv2.INTER_LINEAR)

    if ExTrackbars:
        minDisparity = cv2.getTrackbarPos('minDisparity', 'Parameters Tuner')  # def 27
        numDisparities = cv2.getTrackbarPos('numDisparities', 'Parameters Tuner') * 16  # def 2
        blockSize = cv2.getTrackbarPos('blockSize', 'Parameters Tuner')  # def 3
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'Parameters Tuner')  # def 7
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'Parameters Tuner')  # def 3
        speckleRange = cv2.getTrackbarPos('speckleRange', 'Parameters Tuner')  # def 32
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'Parameters Tuner')  # def 12
        trackbarP1 = cv2.getTrackbarPos('P1', 'Parameters Tuner') * 21 * blockSize * blockSize  # def 8
        trackbarP2 = cv2.getTrackbarPos('P2', 'Parameters Tuner') * 21 * blockSize * blockSize  # def 32
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'Parameters Tuner')  # def 63
        vis_mult = cv2.getTrackbarPos('vis_mult', 'Parameters Tuner')  # def 63
        lmbda = cv2.getTrackbarPos('WLS: lambda', 'Parameters Tuner')
        sigma = cv2.getTrackbarPos('WLS: sigma', 'Parameters Tuner')

        left_matcher.setMinDisparity(minDisparity)
        left_matcher.setNumDisparities(numDisparities)
        left_matcher.setBlockSize(blockSize)
        left_matcher.setUniquenessRatio(uniquenessRatio)
        left_matcher.setSpeckleWindowSize(speckleWindowSize)
        left_matcher.setSpeckleRange(speckleRange)
        left_matcher.setDisp12MaxDiff(disp12MaxDiff)
        left_matcher.setP1(trackbarP1)
        left_matcher.setP2(trackbarP2)
        left_matcher.setPreFilterCap(preFilterCap)
    else:
        numDisparities = cv2.getTrackbarPos('numDisparities', 'Parameters Tuner') * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'Parameters Tuner')
        lmbda = cv2.getTrackbarPos('WLS: lambda', 'Parameters Tuner')
        sigma = cv2.getTrackbarPos('WLS: sigma', 'Parameters Tuner')
        vis_mult = cv2.getTrackbarPos('vis_mult', 'Parameters Tuner')
        left_matcher.setNumDisparities(numDisparities)
        left_matcher.setBlockSize(blockSize)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    if ExTrackbars:
        right_matcher.setMinDisparity(minDisparity)
        right_matcher.setNumDisparities(numDisparities)
        right_matcher.setBlockSize(blockSize)
        right_matcher.setSpeckleWindowSize(speckleWindowSize)
        right_matcher.setSpeckleRange(speckleRange)
        right_matcher.setDisp12MaxDiff(disp12MaxDiff)
    else:
        right_matcher.setNumDisparities(numDisparities)
        right_matcher.setBlockSize(blockSize)

    left_disparity = left_matcher.compute(imLrecti, imRrecti)
    right_disparity = right_matcher.compute(imRrecti, imLrecti)
    converted_disparity = convert_disp(left_disparity)
    converted_disparity_right = convert_disp(right_disparity)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disparity = wls_filter.filter(converted_disparity, imLrecti, disparity_map_right=converted_disparity_right)
    filtered_disparity_CV16S = cv2.normalize(filtered_disparity, None, alpha=-32768, beta=32767,
                                             norm_type=cv2.NORM_MINMAX, \
                                             dtype=cv2.CV_16S)

    disparity_vis = cv2.ximgproc.getDisparityVis(filtered_disparity_CV16S,
                                                 vis_mult)  # type of disparity_vis -> UINT 8 (0-255)

    depth_map = np.zeros(np.shape(filtered_disparity))
    depth_map = focal_length_cam0 * baseline / filtered_disparity
    mask = cv2.inRange(depth_map, 0, depth_threshold)

    # Number of active pixels bigger than 1% of the pixel sum
    if np.sum(mask) / 255.0 > 0.01 * mask.shape[0] * mask.shape[1]:
        # Contour detection
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)

        # Check if detected contour is significantly large (to avoid multiple tiny regions)
        if cv2.contourArea(cnts[0]) > 0.01 * mask.shape[0] * mask.shape[1]:
            x, y, w, h = cv2.boundingRect(cnts[0])
            # Finding average depth of region represented by the largest contour
            mask2 = np.zeros_like(mask)
            cv2.drawContours(mask2, cnts, 0, (255), -1)
            # Calculating the average depth of the object closer than the safe distance
            valid_values = depth_map[np.where(np.logical_and(depth_map != np.inf, mask2 == 255))]
            depth_mean = np.mean(valid_values)
            centerX = (x + w) // 2
            centerY = (y + h) // 2
            cv2.putText(imLrecti, "WARNING !", (centerX, centerY), 1, 2, (0, 0, 255), 2, 2)
            cv2.putText(imLrecti, "Object at", (centerX, centerY + 30), 1, 2, (0, 255, 0), 2, 2)
            cv2.putText(imLrecti, "%.2f m" % depth_mean, (centerX, centerY + 60), 1, 2, (0, 255, 0), 2, 2)
            cv2.namedWindow('Mask2')
            cv2.rectangle(imLrecti, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow('Mask2', mask2)
        else:
            cv2.putText(imLrecti, "SAFE!", (100, 100), 1, 3, (0, 255, 0), 2, 3)

    cv2.namedWindow('left image')
    cv2.imshow('left image', imLrecti)

    cv2.namedWindow('mask')
    cv2.imshow('mask', mask)

    cv2.namedWindow('depth_map')
    cv2.imshow('depth_map', depth_map)
    cv2.setMouseCallback("depth_map", mouse_callback)

    if cv2.waitKey(1) == 27:
        switch += 1

    if switch % 2 == 0:  # Pause
        pass
    else:
        i = i + 1

    if i == len(gt_disparity_map) - 1:  # Rewind to the beginning
        i = 1
