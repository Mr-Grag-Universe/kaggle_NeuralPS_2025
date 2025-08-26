import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

def get_bounds(data, find_deviation_point):
    # always returns numpy!!!
    assert len(data.shape) == 1, RuntimeError('data must be 1d array')
    data_pt = torch.tensor(data)
    data = data_pt.detach().numpy()
    smoothed = gaussian_filter1d(data, sigma=3)
    
    first_derivative = np.gradient(smoothed)
    second_derivative = np.gradient(first_derivative)
    
    valley_left = np.argmin(first_derivative)
    valley_right = np.argmax(first_derivative)
    return find_deviation_point(second_derivative, valley_left, valley_right)

def get_transit_bounds_external(data):
    def find_min_deviation_point(second_derivative, l, r):
        left_half = second_derivative[:l]
        left_bound = 0 if (len(left_half) < 2) else np.argmin(left_half)
        local_left_boundary = left_bound
        
        right_half = second_derivative[r:]
        right_bound = 0 if (len(right_half) < 2) else np.argmin(right_half)
        local_right_boundary = r + right_bound

        return local_left_boundary, local_right_boundary
    
    return get_bounds(data, find_min_deviation_point)

def get_transit_bounds_internal(data):
    def find_max_deviation_point(second_derivative, l, r):
        c = (l+r)//2
        
        left_half = second_derivative[l:c]
        left_bound = 0 if (len(left_half) < 2) else np.argmax(np.abs(left_half))
        local_left_boundary = l + left_bound
        
        right_half = second_derivative[c:r]
        right_bound = 0 if (len(right_half) < 2) else np.argmax(np.abs(right_half))
        local_right_boundary = c + right_bound

        return local_left_boundary, local_right_boundary
        
    return get_bounds(data, find_max_deviation_point)

