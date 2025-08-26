import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

def get_bounds(data, find_deviation_point):
    # always returns numpy!!!
    data_pt = torch.tensor(data)
    data = data_pt.detach().numpy()
    smoothed = gaussian_filter1d(data, sigma=3)
    
    first_derivative = np.gradient(smoothed)
    second_derivative = np.gradient(first_derivative)
    
    valley_left = np.argmin(first_derivative)
    valley_right = np.argmax(first_derivative)
    valley_center = (valley_left + valley_right) // 2

    left_half = second_derivative[valley_left:valley_center]
    local_left_boundary = valley_left+find_deviation_point(left_half)
    right_half = second_derivative[valley_center:valley_right]
    local_right_boundary = valley_center + find_deviation_point(right_half)
    
    return local_left_boundary, local_right_boundary

def get_transit_bounds_external(data):    
    def find_min_deviation_point(second_derivative):
        if len(second_derivative) < 2:
            return 0
        return np.argmin(np.abs(second_derivative))
    return get_bounds(data, find_min_deviation_point)

def get_transit_bounds_internal(data):    
    def find_max_deviation_point(second_derivative):
        if len(second_derivative) < 2:
            return 0
        return np.argmax(np.abs(second_derivative))
    return get_bounds(data, find_max_deviation_point)
