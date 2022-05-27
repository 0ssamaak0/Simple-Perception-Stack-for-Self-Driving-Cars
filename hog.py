# importing the libraris
import numpy as np
import glob
import matplotlib.pyplot as plt
import joblib
import time
import cv2
import sys

# importing the functions
from window import *
from features_functions import *
from SVM import *

cars = []
notcars = []

cars_path = glob.glob('smaller/vehicles/*/*.jpeg')
notcars_path = glob.glob('smaller/non-vehicles/*/*.jpeg')

for image in notcars_path:
    notcars.append(image)
for image in cars_path:
    cars.append(image)


# parameters
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hist_bins = 16    # Number of histogram bins

train = False
svm_pkl = "hog/SVC.pkl"
scaler_pkl = "hog/X_Scaler.pkl"

if train:
    car_features = extract_features(cars, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)

    notcar_features = extract_features(notcars, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
    print("--feature extraction done--")

    svc, X_scaler = SVM(car_features, notcar_features)
    print("--training done--")

    joblib.dump(svc, svm_pkl)
    joblib.dump(X_scaler, scaler_pkl)
else:
    svc = joblib.load(svm_pkl)
    X_scaler = joblib.load(scaler_pkl)

# We will make different window sizes for different sizes of cars
# We will make different window sizes for different
sizes_and_ystarts = [[40, [400, 80]],
                     [80, [400, 520]],
                     [120, [400, 400 + 180]],
                     [160, [400, 400 + 160 + 80]],
                     [200, [400, None]],
                     [240, [400, None]],
                     [300, [300, None]],
                     [340, [300, None]]]


# image = plt.imread("bbox-example-image.jpg")
input_path = sys.argv[1]
output_path = sys.argv[2]


capture = cv2.VideoCapture(input_path)
isTrue, frame = capture.read()

fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

writer = cv2.VideoWriter(output_path, fourcc, fps,
                         (frame.shape[1], frame.shape[0]))

i = 0
save = i
t1 = time.time()
while i < frame_count:
    isTrue, frame = capture.read()
    if not isTrue:
        try:
            print(percentage + "OOOOOOOOOOOOOO\nO\n")
        except:
            print("\nFinished")
        break
    hot_windows = []
    for size_and_ystart in sizes_and_ystarts:

        windows = slide_window(frame, x_start_stop=[None, None], y_start_stop=[size_and_ystart[1][0], size_and_ystart[1][1]],
                               xy_window=(size_and_ystart[0], size_and_ystart[0]), xy_overlap=(0.5, 0.5))

        result = search_windows(frame, windows, svc, X_scaler, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block)

        for res in result:
            hot_windows.append(res)

    labels = heatmap_thresh(frame, hot_windows, 1)
    final_img = draw_labeled_bboxes(frame, labels)

    i += 1
    if i < save:
        i = save + 1
    save = i

    t2 = divmod(time.time() - t1, 60)

    mins = round(t2[0])
    if mins < 10:
        mins = "0" + str(mins)

    secs = round(t2[1])
    if secs < 10:
        secs = "0" + str(secs)

    percentage = round(((i * 100 / fps) / duration), 1)

    loading = ("■" * int(percentage / 2)) + ("□" * (50 - int(percentage)))

    sys.stdout.write(f"\r{percentage}% time:{mins}:{secs} {loading}")
    sys.stdout.flush()
    time.sleep(0.01)

    writer.write(final_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
