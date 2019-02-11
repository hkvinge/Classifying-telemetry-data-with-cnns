import numpy as np
import pywt
from scipy import signal


def convolution(ts,dims,widths):
    # Function to perform convolutions
 
    ts_image = np.zeros((dims[0],widths+1))
    
    # Choose a width for wavelets
    widths = np.arange(1,widths+1)
    
    # Calculate wavelet image
    ts_image = pywt.cwt(ts, widths, 'mexh')
    
    ts_image = ts_image[0]
    
    return ts_image

def create_image(time_series,length,widths,channels):
   
    # Initialize arrays to store "images" and their normalization
    image = np.zeros((widths,length,channels))
    image_normalized = np.zeros((widths,length,channels))
    
    # Loop through channels and create an image for each channel
    for i in range(channels):
        image[:,:,i] = convolution(time_series,(length,1),widths)
    
    return image
