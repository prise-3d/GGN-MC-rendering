import numpy as np
from ipfml.processing import reconstruction

from scipy.ndimage import gaussian_filter
from ipfml.processing.transform import get_LAB_L
import cv2

def compute_svd_reconstruction(data):

    # b_size, _, _, _ = data.shape

    # batch_data = []

    # for i in range(b_size):
        # compute svd reconstruction for each sequence image of data
        # then compute abs difference from predicted AE reference

    current_predicted = data #.transpose(1, 2, 0)

    h, w, _ = current_predicted.shape

    predicted_first_part = reconstruction.svd(current_predicted, (int(h / 4), h))
    predicted_middle_part = reconstruction.svd(current_predicted, (int(h / 2), h))
    predicted_last_part = reconstruction.svd(current_predicted, (int(h / 2) + int(h / 4), h))

    predicted_svd_reconstruct = np.array([predicted_first_part, predicted_middle_part, predicted_last_part])
    # batch_data.append(predicted_svd_reconstruct)

    return predicted_svd_reconstruct

def compute_noise_mask(data):

    current_data = data #[i].transpose(1, 2, 0)
    lab_data = get_LAB_L(current_data) #/ 100. # max value for lab color space is 100.

    # gaussian filter with sigma \in [0.3, 1.5]
    # filtered_data = gaussian_filter(lab_data, sigma=0.3)
    # filtered_data = gaussian_filter(filtered_data, sigma=1.5)
    filtered_data = cv2.GaussianBlur(lab_data, (3, 3), 0.3, 0.3)
    filtered_data = cv2.GaussianBlur(filtered_data, (3, 3), 1.5, 1.5)

    noise_mask = np.abs(lab_data - filtered_data)
    
    # normalize the noise mask data
    # avoid division by zero
    # if np.max(noise_mask) > 100.:
    # output_mask = noise_mask / (np.max(noise_mask) + 0.000000001)
    output_mask = noise_mask / 100.  # same scale for data

    return output_mask