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
length = 2400
widths = 50
channels = 1
step_size = 50

# Path to data files
path_to_data = "/Users/HK/Programming/Calcom/tamu"

# Path to directory where network data will be stored
path_to_store_weights = "/Users/HK/Mathematics/Projects/time_series_pictures/weights"

# Start time
t0 = time.time()

# Use Calcom functionality to load time series    
temps = utils.load_all(window=length);
#acel = utils.load_all(window=length,which='a')
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
data_total = np.concatenate((labels,temps),axis=1)
# Now shuffle total array
np.random.seed(0)
np.random.shuffle(data_total)
# Separate labels and features
labels_subsampled = data_total[0:numb_train + numb_test,0]
labels = labels_subsampled
temps_subsampled = data_total[0:numb_train + numb_test,1:numb_train + numb_test:step_size]
for i in range(1,step_size):
    offset = data_total[0:numb_train + numb_test,i::step_size]
    temps_subsampled = np.concatenate((temps_subsampled,offset),axis=0)
    labels = np.concatenate((labels,labels_subsampled),axis=0)
temps = temps_subsampled
length = int(length/step_size)

# Separate training and testing data and take transpose
labels_train = labels[0:step_size*numb_train]
labels_train = labels_train.transpose()
labels_test = labels[step_size*numb_train:step_size*numb_ex]
labels_test = labels_test.transpose()
temps_train = temps[0:step_size*numb_train,:]
temps_train = temps_train.transpose()
temps_test = temps[step_size*numb_train:step_size*numb_ex,:]
temps_test = temps_test.transpose()

# Initialize array to hold training images
images_train = np.zeros((widths, length, channels, step_size*numb_train))
# Initialize array to hold evaluation images
images_test = np.zeros((widths, length, channels, step_size*numb_test))
    
# Iterate through, setting desired number of training images
for i in range(step_size*numb_train):
    time_series = temps_train[:,i]
    images_train[:,:,:,i] = create_image(time_series,length,widths,channels)
    if labels_train[i] > 0:
        labels_train[i] = 1  
    if (i % 100 == 0):
        print(str(i) + " out of " + str(step_size*numb_train) + " training images.")  

# Now mean center each pixel for training images
for i in range(length):
    for j in range(widths):
        mean = np.mean(images_train[j,i,:,:])
        variance = np.var(images_train[j,i,:,:])
        images_train[j,i,:,:] = images_train[j,i,:,:] - mean*np.ones((1,1,channels,step_size*numb_train))
        images_train[j,i,:,:] = (1/variance)*images_train[j,i,:,:]

# Iterate through, number of test images
for i in range(step_size*numb_test):
    time_series = temps_test[:,i]
    images_test[:,:,:,i] = create_image(time_series,length,widths,channels)
    if labels_test[i] > 0:
        labels_test[i] = 1            
    if (i % 100 == 0):
        print(str(i) + " out of " + str(step_size*numb_test) + " test images.") 
 
# Now mean center each pixel for test images
for i in range(length):
    for j in range(widths):
        mean = np.mean(images_test[j,i,:,:])
        variance = np.var(images_test[j,i,:,:])
        images_test[j,i,:,:] = images_test[j,i,:,:] - mean*np.ones((1,1,channels,step_size*numb_test))
        images_test[j,i,:,:] = (1/variance)*images_test[j,i,:,:]

# Plot some example time series colored by label
# X coordinate
x = range(length)
plt.plot(x,temps_train[:,1], label=labels_train[1])
plt.plot(x,temps_train[:,10], label=labels_train[10])
plt.plot(x,temps_train[:,100], label=labels_train[100])
plt.plot(x,temps_train[:,150], label=labels_train[150])
plt.legend()
plt.show()
# And their corresponding images
plt.imshow(images_train[:,:,0,1], interpolation='nearest')
plt.show()
plt.imshow(images_train[:,:,0,10]-images_train[:,:,0,1], interpolation='nearest')
plt.show()
plt.imshow(images_train[:,:,0,100]-images_train[:,:,0,1], interpolation='nearest')
plt.show()
plt.imshow(images_train[:,:,0,150]-images_train[:,:,0,1], interpolation='nearest')
plt.show() 

#Transpose images for input into tensorflow
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
