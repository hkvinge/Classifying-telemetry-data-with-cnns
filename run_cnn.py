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

tf.logging.set_verbosity(tf.logging.INFO)

# Set length of time interval to investigate
length = 64
widths = 32
channels = 3
numb_train = 50000
numb_eval = 2000

# Path to data files
path_to_data = # Place path here

# Path to directory where network data will be stored
path_to_store_weights = # Place path here

# Start time
t0 = time.time()
    
# Initialize variable to hold data
acel = np.zeros(0)
    
# Read in data from CSV file for training
for i in range(1,13):
    temp = genfromtxt(path_to_data +str(i)+'.csv', delimiter=',')
    if (i == 1):
        acel = temp
    else:
        acel = np.concatenate((acel,temp))

# Read in data from CSV file for testing
for i in range(13,16):
    temp = genfromtxt(path_to_data +str(i)+'.csv', delimiter=',')
    if (i == 13):
        acel_eval = temp
    else:
        acel_eval = np.concatenate((acel_eval,temp))


# Get total length of time series
dims = np.shape(acel)
dims_eval = np.shape(acel_eval)
    
# Initialize array for corresponding labels
training_labels = np.zeros(numb_train)
    
# Initialize array for corresponding labels
eval_labels = np.zeros(numb_eval)
    
# Initialize array to hold training images
training_images = np.zeros((widths, length, channels, numb_train))
# Initialize array to hold evaluation images
eval_images = np.zeros((widths, length, channels, numb_eval))
    
# Variable to check how many training examples have been added
count = 0
    
# Iterate through desired, setting desired number of training images
while count < numb_train:
        
    # Generate a random integer in proper range
    random_integer = randint(0, dims[0]-length)
        
    # Check whether all labels on the time series are same
    labels = acel[random_integer:random_integer + length,4]
    unique_elements = np.unique(labels).size
        
        
    if (unique_elements == 1) and (acel[random_integer,4] != 0):
        # Assign training images
        training_images[:,:,:,count] = create_image(acel,length,random_integer,widths,channels)
            
        # Add in label
        training_labels[count] = acel[random_integer,4]
            
        # Iterate count
        count = count + 1


count = 0
    
# Iterate through desired number of test images
while count < numb_eval:
        
    # Generate a random integer in proper range
    random_integer = randint(0, dims_eval[0]-length)
        
    # Check whether all labels on the time series are same
    labels = acel_eval[random_integer:random_integer + length,4]
    unique_elements = np.unique(labels).size
        
    if (unique_elements == 1) and (acel_eval[random_integer,4] != 0):
        # Assign training images
        eval_images[:,:,:,count] = create_image(acel_eval,length,random_integer,widths,channels)
            
        # Add in label
        eval_labels[count] = acel_eval[random_integer,4]
            
        # Iterate count
        count = count + 1

# Shift labels on training set to be between 0 and 7
training_labels = training_labels - np.ones(np.shape(training_labels))
eval_labels = eval_labels - np.ones(np.shape(eval_labels))
    
# Transpose images for input into tensorflow
training_images = np.transpose(training_images)
eval_images = np.transpose(eval_images)
    
# Cast arrays from float64 to float32 for tensorflow
training_images = np.float32(training_images)
training_labels = training_labels.astype(np.int32)
eval_images = np.float32(eval_images)
eval_labels = eval_labels.astype(np.int32)
    
# Create estimator
time_series_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir=path_to_store_weights)
    
# Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
tensors_to_log = {}
logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)
                                                    
# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":training_images},
            y=training_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
                                                    
time_series_classifier.train(
            input_fn=train_input_fn,
            steps=1000,
            hooks=[logging_hook])
                                                    
# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_images},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
eval_results = time_series_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
