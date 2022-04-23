from this import d
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def image_reader(image):
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


def Canny(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur_gray = cv.GaussianBlur(gray, (3, 3), 0)
    edges = cv.Canny(blur_gray, 50, 150)
    return edges


def cutting(image, edges):
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    vertices = np.array([[(120, image.shape[0]), (410, 330), (550, 330),
                        (image.shape[1] - 80, image.shape[0])]], dtype=np.int32)

    cv.fillPoly(mask, vertices, ignore_mask_color)

    masked_edges = cv.bitwise_and(edges, mask)

    line_image = np.zeros_like(image)

    lines = cv.HoughLinesP(masked_edges, 2, np.pi / 180,
                           15, np.array([]), 40, 20)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)

    lines_on_road = cv.addWeighted(line_image, 0.8, image, 1, 0)
    return lines_on_road
