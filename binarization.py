import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def S_thresholder(S, thresh=(0, 255)):

    S_threshold = S * 0
    S_threshold[(S >= thresh[0]) & (S <= thresh[1])] = 1

    return S_threshold


def binarization_choice(img):
    S = cv.cvtColor(img, cv.COLOR_BGR2HLS)[:, :, 2]

    sthresh = S_thresholder(S, (90, 255)) * 255
    canny = cv.Canny(cv.GaussianBlur(S, (5, 5), 0), 50, 170)

    binary_sobx_Sthreshs = S * 0
    binary_sobx_Sthreshs[(sthresh == 255) | (canny == 255)] = 255

    return binary_sobx_Sthreshs
