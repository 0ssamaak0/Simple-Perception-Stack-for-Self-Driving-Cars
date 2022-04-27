import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def S_thresholder(S, thresh=(0, 255)):

    S_threshold = S * 0
    S_threshold[(S >= thresh[0]) & (S <= thresh[1])] = 1

    return S_threshold


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == "x":
        sobel = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == "y":
        sobel = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.abs(sobel)

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    sbinary = scaled_sobel * 0
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel < thresh[1])] = 1

    return sbinary


# bad one
def binarization_choice1(img):
    S = cv.cvtColor(img, cv.COLOR_BGR2HLS)[:, :, 2]

    sthresh = S_thresholder(S, (90, 255)) * 255
    canny = cv.Canny(cv.GaussianBlur(S, (5, 5), 0), 50, 170)

    binary_sobx_Sthreshs = S * 0
    binary_sobx_Sthreshs[(sthresh == 255) | (canny == 255)] = 255

    return binary_sobx_Sthreshs


def BGR_equlization(frame, B=255, G=255, R=255):
    B_eq = np.uint8(cv.equalizeHist(frame[:, :, 0]) * (B / 255))
    G_eq = np.uint8(cv.equalizeHist(frame[:, :, 1]) * (G / 255))
    R_eq = np.uint8(cv.equalizeHist(frame[:, :, 2]) * (R / 255))

    return cv.merge((B_eq, G_eq, R_eq))


def HLS_equlization(frame, H=255, L=255, S=255):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    H_eq = np.uint8(cv.equalizeHist(frame[:, :, 0]) * (H / 255))
    L_eq = np.uint8(cv.equalizeHist(frame[:, :, 1]) * (L / 255))
    S_eq = np.uint8(cv.equalizeHist(frame[:, :, 2]) * (S / 255))

    return cv.cvtColor(cv.merge((H_eq, L_eq, S_eq)), cv.COLOR_HLS2BGR)


# current
def binarization_choice2(frame):
    frame_equlized_HLS = HLS_equlization(frame, 100, 50, 255)

    S = cv.cvtColor(frame_equlized_HLS, cv.COLOR_BGR2HLS)[:, :, 2]

    sthresh = S_thresholder(S, (140, 230)) * 255

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    canny = cv.Canny(gray, 40, 80)

    gaussian = cv.GaussianBlur(gray, (9, 9), 0)
    sobelx = abs_sobel_thresh(gaussian, "x", 3, (40, 220)) * 255

    sobelx[canny == 255] = 0
    sobelx = cv.dilate(sobelx, (15, 15))

    binary = S * 0
    binary[(sthresh == 255) | (sobelx == 255)] = 255

    closing = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((7, 7)))

    return closing
