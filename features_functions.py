import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog

# Hog features function


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False):
    if vis:
        hog_features, hog_image = hog(img, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      visualize=vis, feature_vector=feature_vec,
                                      block_norm="L2-Hys")
        return hog_features, hog_image
    else:
        hog_features = hog(img, orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           visualize=vis, feature_vector=feature_vec,
                           block_norm="L2-Hys")
        return hog_features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # bin_edges = rhist[1]

    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    return hist_features


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified) N/A
    # Use cv2.resize().ravel() to create the feature vector
    small_image = cv2.resize(img, (size))
    # Return the feature vector
    return small_image.ravel()


def extract_SH_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in imgs:
        # Read in each one by one
        img = plt.imread(img)
        # apply color conversion if other than 'RGB' (N/A)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(img, size=spatial_size)
        # Apply color_hist() to get color histogram features
        hist_features = color_hist(img, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        # concat = np.concatenate((spatial_features, hist_features))
        # if(features.shape[0] == 0):
        #     features = concat
        # else:
        #     features = np.vstack((features, concat))
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features


def extract_features(imgs,
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2,
                     ):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in imgs:
        # Read in each one by one
        img = plt.imread(img)
        feature_image = np.copy(img)

        hog_features = []

        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:, :, channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)

        hist_features = color_hist(img, nbins=hist_bins)

        features.append(np.concatenate((hog_features, hist_features)))
    return features


def single_img_features(img,
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2,
                        ):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    # Read in each one by one
    # img = plt.imread(img)
    feature_image = np.copy(img)
    # Call get_hog_features() with vis=False, feature_vec=True
    hog_features = []
    for channel in range(feature_image.shape[2]):
        hog_features.append(get_hog_features(feature_image[:, :, channel],
                            orient, pix_per_cell, cell_per_block,
                            vis=False, feature_vec=True))
    hog_features = np.ravel(hog_features)
    # Append the new feature vector to the features list
    # apply color conversion if other than 'RGB' (N/A)
    # Apply bin_spatial() to get spatial color features
# spatial_features = bin_spatial(img, size=spatial_size)
    # Apply color_hist() to get color histogram features
    hist_features = color_hist(img, nbins=hist_bins)
    # Append the new feature vector to the features list
    # concat = np.concatenate((spatial_features, hist_features))
    # if(features.shape[0] == 0):
    #     features = concat
    # else:
    #     features = np.vstack((features, concat))
    features.append(np.concatenate((hog_features, hist_features)))
    return features
