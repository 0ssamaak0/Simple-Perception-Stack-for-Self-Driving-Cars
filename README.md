# Simple Perception Stack for Self-Driving Cars

output video must be mp4

``` bash
> python3 project_main.py input_path output_path debugging
```
example:
``` bash
> python3 project_main.py project_video.mp4 result.mp4 0
```
# Phase 1: Lane Detection
...................


# Phase 2: Object Detection (Vehicles)
in this phase, we want to detect the vehicles through given videos

## Training Pipeline
### 1. Getting the dataset
In [hog.py](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/2459aa39a461406a2f2df4b045532e8c6bfafec3/hog.py#L18) file, we get the datasets paths for **vehicles** and **non-vehicles** and store them into 2 lists

```python
cars_path = glob.glob('path to vehicles')
notcars_path = glob.glob('path to non-vehicles')

for image in notcars_path:
    notcars_image_list.append(image)
for image in cars_path:
    cars_image_list.append(image)
```
### 2.Parameters
since we will use only **hog features** and **Color_histogram** features, we have 4 parameters

```python
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hist_bins = 16    # Number of histogram bins
```

### 3.Feature extraction
using [extract_features()](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/2459aa39a461406a2f2df4b045532e8c6bfafec3/features_functions.py#L70) function, we will extract the feautres of both **vehicles** and **non-vehicles** and store them into two lists.

the [extract_features()](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/2459aa39a461406a2f2df4b045532e8c6bfafec3/features_functions.py#L70) function takes the **image_list** we have prepared previously and the determined parameters as following

```python
features = extract_features(image_list, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
``` 
#### 1.hog features
using [get_hog_features()](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/2459aa39a461406a2f2df4b045532e8c6bfafec3/features_functions.py#L9) and we have chosen to get the hog features of the three RGB Channels.

the function takes the image and the hog parameters, we are able to get the **hog_features** only, or a 2 element tuple containing the **hog_features** and the **hog_image**

```python
hog_features.append(get_hog_features(feature_image[:, :, channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=True))
```
![hog_image](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/master/Images/hog_image.png)

#### 2.color histogram features
we use [color_hist()](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/2459aa39a461406a2f2df4b045532e8c6bfafec3/features_functions.py#L26) function to get the RGB histograms of the image, the function returns them concatenated into single array

![color_hist](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/master/Images/color_hist.png)

### 4.Training the model
after scaling the training set, we have trained an [SVM](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/master/SVM.py) using the full dataset (since we will test on videos and images later)

the SVM function returns the **trained model** and the **scaler** to scale the test images later


### 5.Pickling
after training the model successfully, we pickled the results to use them later in testing without retraining each time
```python
svc, X_scaler = SVM(car_features, notcar_features)

joblib.dump(svc, svm_pkl)
joblib.dump(X_scaler, scaler_pkl)
```

## Testing Pipeline

### 1.sliding windows
in [slide_window()](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/25a4ace846b4baa1e2e98c960eb7dc371cf23eb2/window.py#L7) we take the test image, sizes of the windows and their positions, overlapping if needed and the size of the small window as following
```python
windows = slide_window(frame, x_start_stop=[None, None], y_start_stop=[size_and_ystart[1][0], size_and_ystart[1][1]],
                               xy_window=(size_and_ystart[0], size_and_ystart[0]), xy_overlap=(0.5, 0.5))
```

The image is now divided into windows (will be used later to search within)

![window_image](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/master/Images/image_windows.png)


### 2.Searching within the window
we pass each window list to [search_windows()](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/25a4ace846b4baa1e2e98c960eb7dc371cf23eb2/window.py#L62) function, then we resize the test image to match the training images size, and use [single_img_features()](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/2459aa39a461406a2f2df4b045532e8c6bfafec3/features_functions.py#L96) to extract the same features extracted from the training images

we use the **classifier** and the **scaler** to test the image, if the prediction is 1, we append this window and return it

through the iterations we fill the initially empty list [hot_windows](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/2459aa39a461406a2f2df4b045532e8c6bfafec3/hog.py#L92) these are the windows detecting the cars, but unfortunately they may have duplicates and false positives

### 3.heatmap and thresholding
in [heatmap_thresh()](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/25a4ace846b4baa1e2e98c960eb7dc371cf23eb2/window.py#L91) function we take the image and the **hot_windows** and a threshold

making an empty heatmap, then we fill it with each window, usually some windows overlap, and some don't, we take the overlapped images in which the overlap is higher than a certain threshold.

this function returns a 2D tuple that contains the positions of the boxes (else is zeros) and their numbers

### 4.final result
we construct the final boxes using [draw_labeled_bboxes()](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/25a4ace846b4baa1e2e98c960eb7dc371cf23eb2/window.py#L107) function which only takes the positions of the thresholded heatmap, we take the nonzero values and iteratie through them using the index of the box and draw each box in its position


![heatmap_image](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/master/Images/heatmap.png)