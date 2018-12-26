import numpy as np
import pywt
from scipy import signal


def convolution(ts,dims,widths):
    # Function to perform convolutions
    
    # Interpolate missing data
    ts = linearly_interpolate_nans(ts)
    ts_image = np.zeros((dims[0],widths+1))
    
    # Choose a width for wavelets
    widths = np.arange(1,widths+1)
    
    #print(pywt.wavelist(kind='continuous'))
    
    # Calculate wavelet image
    ts_image = pywt.cwt(ts, widths, 'mexh')
    
    #print(ts_image)
    
    ts_image = ts_image[0]
    
    #plt.imshow(ts_image)
    #plt.show()
    
    # Return image
    return ts_image

def linearly_interpolate_nans(y):
    # Fit a linear regression to the non-nan y values
    
    # Create X matrix for linreg with an intercept and an index
    X = np.vstack((np.ones(len(y)), np.arange(len(y))))
    
    # Get the non-NaN values of X and y
    X_fit = X[:, ~np.isnan(y)]
    y_fit = y[~np.isnan(y)].reshape(-1, 1)
    
    # Estimate the coefficients of the linear regression
    beta = np.linalg.lstsq(X_fit.T, y_fit)[0]
    
    # Fill in all the nan values using the predicted coefficients
    y.flat[np.isnan(y)] = np.dot(X[:, np.isnan(y)].T, beta)
    return y

def create_image(time_series,length,start,widths,channels):
    
    # Initialize arrays to store "images" and their normalization
    image = np.zeros((widths,length,channels))
    image_normalized = np.zeros((widths,length,channels))
    
    # Loop through channels and create an image for each channel
    for i in range(channels):
        image[:,:,i] = convolution(time_series[start:start + length,i],(length,1),widths)
    
    return image
