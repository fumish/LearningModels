"""
This is module for kernels
"""
from scipy.spatial import distance_matrix
import numpy as np

def gauss_kernel(x:np.ndarray, y:np.ndarray, theta1:float=1, theta2:float=1):
    return theta1 * np.exp(-distance_matrix(x,y,p=2)**2 / theta2)

def exp_kernel(x:np.ndarray, y:np.ndarray, theta1:float=1, theta2:float=1):
    return theta1 * np.exp(-distance_matrix(x,y,p=1) / theta2)

def add_kernel(kernel1, kernel2, x:np.ndarray, y:np.ndarray, theta1:float=1, theta2:float=2):
    return kernel1(x,y,theta1,theta2) + kernel2(x,y,theta1,theta2)
