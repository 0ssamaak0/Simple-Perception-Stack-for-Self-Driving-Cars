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
### Getting the dataset
In [hog.py](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/2459aa39a461406a2f2df4b045532e8c6bfafec3/hog.py#L18) file, we get the datasets paths for **vehicles** and **non-vehicles** and store them into 2 lists

```python
cars_path = glob.glob('path to vehicles')
notcars_path = glob.glob('path to non-vehicles')

for image in notcars_path:
    notcars_image_list.append(image)
for image in cars_path:
    cars_image_list.append(image)
```
### Parameters
since we will use only **hog features** and **Color_histogram** features, we have 4 parameters

```python
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hist_bins = 16    # Number of histogram bins
```

### Feature extraction
using [extract_features()](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/2459aa39a461406a2f2df4b045532e8c6bfafec3/features_functions.py#L70) function, we will extract the feautres of both **vehicles** and **non-vehicles** and store them into two lists.

the [extract_features()](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/2459aa39a461406a2f2df4b045532e8c6bfafec3/features_functions.py#L70) function takes the **image_list** we have prepared previously and the determined parameters as following

```python
    features = extract_features(**image_list**, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
``` 
#### hog features
using [get_hog_features()](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/2459aa39a461406a2f2df4b045532e8c6bfafec3/features_functions.py#L9) and we have chosen to get the hog features of the three RGB Channels.

the function takes the image and the hog parameters, we are able to get the **hog_features** only, or a 2 element tuple containing the **hog_features** and the **hog_image**

```python
hog_features.append(get_hog_features(feature_image[:, :, channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=True))
```
![hog_image](hog_image_path)
#### color histogram features
we use [color_hist()](https://github.com/0ssamaak0/Simple-Perception-Stack-for-Self-Driving-Cars/blob/2459aa39a461406a2f2df4b045532e8c6bfafec3/features_functions.py#L26) function to get the RGB histograms of the image, the function returns them concatenated into single array

![color_hist](color_hist_path)

### Training the model

