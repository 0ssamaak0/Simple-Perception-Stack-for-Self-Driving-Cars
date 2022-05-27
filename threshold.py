import numpy as np
import cv2 as cv
import cv2


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


def abs_sobel_thresh1(img, orient='x', thresh_min=0, thresh_max=255, fullimage=True):
    # Apply the following steps to img
    # 1) Convert to grayscale
    if fullimage:
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        grey = img
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        # Here dx = 1 and dy = 0
        gradient = cv2.Sobel(grey, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        # Here dx = 0 and dy = 1
        gradient = cv2.Sobel(grey, cv2.CV_64F, 0, 1)
    # For the gradient, the range of output will be from -4*255 to 4*255
    # 3) Take the absolute value of the derivative or gradient, now the range will be from 0 to 4*255
    gradient_abs = abs(gradient)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_gradient_abs = np.uint8(255*gradient_abs/gradient_abs.max())  # if maximum is 4*255 it will be like dividing by 4
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_img = np.zeros_like(scaled_gradient_abs)
    binary_img[(scaled_gradient_abs >= thresh_min) & (scaled_gradient_abs <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    return binary_img


def dir_threshold(img, sobel_kernel=3, thresh=(0.85, 1.05)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def threshold(image):
    output_sobelx = abs_sobel_thresh1(image, orient='x', thresh_min=20, thresh_max=100)
    output_sobely = abs_sobel_thresh1(image, orient='y', thresh_min=20, thresh_max=100)
    output_dir = dir_threshold(image, sobel_kernel=5, thresh=(0.8, 1.3))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_hsv = hsv[:, :, 1]
    h_hsv = hsv[:, :, 0]
    v_hsv = hsv[:, :, 2]

    combined = np.zeros_like(s_hsv)
    edge_mask = (output_sobely == 1) | (output_sobelx == 1) & (output_dir == 1)
    mask = (s_hsv >= 60)
    mask_unwanted = (v_hsv <= 60)
    combined[edge_mask] = 1
    combined[mask] = 1
    combined[mask_unwanted] = 0
    return combined
