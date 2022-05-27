import numpy as np


def draw_rectangle(image, left_eqn, right_eqn):
    line_image = np.copy(image)*0  # creating a blank to draw lines on
    #ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    XX, YY = np.meshgrid(np.arange(0, image.shape[1]), np.arange(0, image.shape[0]))
    region_thresholds = (XX < (right_eqn[0]*YY**2 + right_eqn[1]*YY + right_eqn[2])) & \
                        (XX > (left_eqn[0]*YY**2 + left_eqn[1]*YY + left_eqn[2]))  # & \
    # (YY < (right_eqn[0]*YY**2 + right_eqn[1]*YY + right_eqn[2])) & \
    #(YY > (left_eqn[0]*YY**2 + left_eqn[1]*YY + left_eqn[2]))

    line_image[region_thresholds] = (0xb9, 0xff, 0x99)  # dcffcc
    return line_image


def measure_curvature_pixels(left_fit, right_fit):
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700  # meters per pixel in x dimension

    y_eval = 720   # bottom of image

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*(left_fit[0]/xm_per_pix)*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*(left_fit[0]/xm_per_pix))
    right_curverad = ((1 + (2*(right_fit[0]/xm_per_pix)*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*(right_fit[0]/xm_per_pix))

    ave_curvature = (left_curverad + right_curverad) / 2

    return ave_curvature
