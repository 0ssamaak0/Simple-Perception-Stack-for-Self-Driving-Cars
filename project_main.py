# relevant imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import sys


def per_transform(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[530, 500], [750, 465], [200, img_size[1]], [1150, 645]])
    offset = 150
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                      [offset, img_size[1]-offset],
                      [img_size[0]-offset, img_size[1]-offset]
                      ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, Minv


def transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)
    return warped


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


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    # print(left_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    # print(binary_warped.shape)
    # print(ploty)
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    for i in range(1, left_fitx.shape[0]):
        cv.line(out_img, (int(left_fitx[i-1]), int(ploty[i-1])),
                (int(left_fitx[i]), int(ploty[i])), (0, 255, 255), thickness=3)
        cv.line(out_img, (int(right_fitx[i-1]), int(ploty[i-1])),
                (int(right_fitx[i]), int(ploty[i])), (0, 255, 255), thickness=3)
    return out_img, left_fit, right_fit


def fit_poly(img_shape, leftx, lefty, rightx, righty, left_fit, right_fit):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fitx, right_fitx, ploty


def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                                                                         left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                                                                           right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(
        binary_warped.shape, leftx, lefty, rightx, righty, left_fit, right_fit)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array(
        [np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    for i in range(1, left_fitx.shape[0]):
        cv.line(result, (int(left_fitx[i-1]), int(ploty[i-1])),
                (int(left_fitx[i]), int(ploty[i])), (0, 255, 255), thickness=3)
        cv.line(result, (int(right_fitx[i-1]), int(ploty[i-1])),
                (int(right_fitx[i]), int(ploty[i])), (0, 255, 255), thickness=3)

    return result, left_fit, right_fit

# Run image through the pipeline
# Note that in your project, you'll also want to feed in the previous fits
# result = search_around_poly(binary_warped)


def draw_rectangle(image, left_eqn, right_eqn):
    line_image = np.copy(image)*0  # creating a blank to draw lines on
    XX, YY = np.meshgrid(
        np.arange(0, image.shape[1]), np.arange(0, image.shape[0]))
    region_thresholds = (YY > (left_eqn[0]*YY**2 + left_eqn[1]*YY + left_eqn[2])) & \
                        (XX > (left_eqn[0]*YY**2 + left_eqn[1]*YY + left_eqn[2])) & \
                        (YY < (right_eqn[0]*YY**2 + right_eqn[1]*YY + right_eqn[2])) & \
                        (XX < (right_eqn[0]*YY**2 +
                         right_eqn[1]*YY + right_eqn[2]))

    line_image[region_thresholds] = (0xb9, 0xff, 0x99)  # dcffcc
    return line_image


# apply the function on videos
# capture is instance of the videocapture class that contains the video given

# input_path = 'project_video_Trim.mp4'
# output_path = "filename_002.mp4"

input_path = sys.argv[1]
output_path = sys.argv[2]

# debugging: 1 for debug mode
debugging = int(sys.argv[3])

capture = cv2.VideoCapture(input_path)

# Video Duration
fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps


isTrue, frame = capture.read()

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

writer = cv2.VideoWriter(output_path, fourcc, 15,
                         (frame.shape[1], frame.shape[0]))

output = binarization_choice2(frame)
warped, m, minv = per_transform(output)
first_time, left_eqn, right_eqn = fit_polynomial(warped)
i = 0
while True:
    isTrue, frame = capture.read()
    if not isTrue:
        print(f"\n the output video have been saved to {output_path} successfully")
        break
    output = binarization_choice2(frame)
    warped = transform(output, m)
    # warped = transform(frame,m)
    output, left_eqn, right_eqn = search_around_poly(
        warped, left_eqn, right_eqn)
    first_time, left_eqn, right_eqn = fit_polynomial(warped)
    rectangle = draw_rectangle(frame, left_eqn, right_eqn)
    correct_rectangle = transform(rectangle, minv)
    transformed_back = transform(output, minv)
    first_stack = cv.addWeighted(transformed_back, 0.5, frame, 1, 0)
    # cv2.imshow('Video', cv.addWeighted(
    #     first_stack, 1, correct_rectangle, 0.7, 0))
    write_frame = cv.addWeighted(first_stack, 1, correct_rectangle, 0.7, 0)
    if debugging:
        # TODO (Debugging mode)
        pass
        # writer.write(The big frame contains the step, refer to the one I used in HP tuner selected)
    else:
        writer.write(write_frame)
    i += 1
    sys.stdout.write("\r%.2f%%" % (i * (10 / 3) / duration))
    sys.stdout.flush()
    # cv2.imshow('Video',warped)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

capture.release()
writer.release()
cv2.destroyAllWindows()
