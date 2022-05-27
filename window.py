import cv2
import numpy as np
from features_functions import single_img_features
from scipy.ndimage.measurements import label


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

    # default size = image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int32(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int32(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int32(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int32(xy_window[1]*(xy_overlap[1]))

    nx_windows = np.int32((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int32((yspan-ny_buffer)/ny_pix_per_step)

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate each window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]

            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
        # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# drawing boxes function
def draw_boxes(img, bboxes, color=(0, 255, 255), thick=6):
    imcopy = img.copy()

    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    return imcopy


def search_windows(img, windows, clf, scaler, hist_bins=32, orient=9,
                   pix_per_cell=8, cell_per_block=2):

    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image and resize to the training image size
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features(), same as extract_features but for 1 image
        features = single_img_features(test_img,
                                       hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block)

        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


""" making a heatmap of the detected objects, then thresholds the number of windows, returns the labels tuple that contains the positions of the objects"""


def heatmap_thresh(img, bbox_list, threshold):
    # empty image
    heatmap = np.zeros_like(img[:, :, 0])
    for box in bbox_list:
        # 1's in the objects places
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # thresholding to remove false positives
    heatmap[heatmap <= threshold] = 0
    # tuple of 2 elements that contains the positions of objects and the number of objects
    labels = label(heatmap)
    return labels


""" draws boxes in the elements higher than the threshold """


def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()

        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 5)

    return img
