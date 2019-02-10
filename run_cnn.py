from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from random import randint
from convolution_functions import*
from cnn import cnn_model_fn
import utils

tf.logging.set_verbosity(tf.logging.INFO)

# Set length of time interval to investigate
length = 1200
widths = 32
channels = 1
step_size = 4

# Path to data files
path_to_data = "/Users/HK/Programming/Calcom/tamu"

# Path to directory where network data will be stored
path_to_store_weights = "/Users/HK/Mathematics/Projects/time_series_pictures/weights"

# Start time
t0 = time.time()

# Use Calcom functionality to load time series    
temps = utils.load_all(window=length);
labels = utils.get_labels('t_post_infection')

# Get split numbers between testing and training
dims = np.shape(temps)
# Number of time series
numb_ex = dims[0]
# Length of each time series
length_ex = dims[1]
# Get size of training set
numb_train = round(.7*numb_ex-1)
numb_test = round(.3*numb_ex-1)

# Transpose temps and labels and then shuffle along first dimension
# so that training/testing set will be new each time

# Start by concatentating the data together (so features and labels
# are shuffled identically)
labels = np.reshape(labels,[-1,1])
temps_total = np.concatenate((labels,temps),axis=1)
# Now shuffle total array
np.random.shuffle(temps_total)
# Separate labels and features
labels = temps_total[0:numb_train + numb_test,0]
temps = temps_total[:,1::step_size]
length = int(length/step_size)

# Separate training and testing data and take transpose
labels_train = labels[0:numb_train]
labels_train = labels_train.transpose()
labels_test = labels[numb_train:numb_ex]
labels_test = labels_test.transpose()
temps_train = temps[0:numb_train,:]
temps_train = temps_train.transpose()
temps_test = temps[numb_train:numb_ex,:]
temps_test = temps_test.transpose()

# Initialize array to hold training images
images_train = np.zeros((widths, length, channels, numb_train))
# Initialize array to hold evaluation images
images_test = np.zeros((widths, length, channels, numb_test))
    
# Iterate through, setting desired number of training images
for i in range(numb_train):
    time_series = temps_train[:,i]
    images_train[:,:,:,i] = create_image(time_series,length,widths,channels)
    if labels_train[i] > 0:
        labels_train[i] = 1   
 
# Iterate through, number of test images
for i in range(numb_test):
    time_series = temps_test[:,i]
    images_test[:,:,:,i] = create_image(time_series,length,widths,channels)
    if labels_test[i] > 0:
        labels_test[i] = 1            

# Transpose images for input into tensorflow
images_train = np.transpose(images_train)
images_test = np.transpose(images_test)

# Cast arrays from float64 to float32 for tensorflow
images_train = np.float32(images_train)
labels_train = labels_train.astype(np.int32)
images_test = np.float32(images_test)
labels_test = labels_test.astype(np.int32)

# Create estimator
time_series_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir=path_to_store_weights)
    
# Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
tensors_to_log = {}
logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)                                        
 
# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":images_train},
            y=labels_train,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
                                                    
time_series_classifier.train(
            input_fn=train_input_fn,
            steps=5000,
            hooks=[logging_hook])
                                                    
# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": images_test},
            y=labels_test,
            num_epochs=1,
            shuffle=False)
eval_results = time_series_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
