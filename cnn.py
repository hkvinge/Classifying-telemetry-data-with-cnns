from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import time
from random import randint

tf.logging.set_verbosity(tf.logging.INFO)

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

def cnn_model_fn(features,labels,mode):
    """Model function for CNN."""
    # This specifies the form of the input
    input_layer = tf.reshape(features["x"],[-1,64,32,3])
    

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    # Dense layer
    pool2_flat = tf.reshape(pool2,[-1,16*8*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=7)

    predictions = {
        # Generative predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add 'softmax_tensor' to the graph. It is used for PREDICT and by the 
        # 'logging_hook'.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=.000001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)   

def main(unused_argv):
    
    # Set length of time interval to investigate
    length = 64
    widths = 32
    channels = 3
    numb_train = 50000
    numb_eval = 2000
    
    # Start time
    t0 = time.time()
    
    # Initialize variable to hold data
    acel = np.zeros(0)

    # Read in data from CSV file for training
    for i in range(1,15):
        temp = genfromtxt('Activity_recognition_from_single_chest-mounted_accelerometer/'+str(3)+'.csv', delimiter=',')
        if (i == 1):
            acel = temp
        else:
            acel = np.concatenate((acel,temp))

    # Read in data from CSV file for testing
    acel_eval = genfromtxt('Activity_recognition_from_single_chest-mounted_accelerometer/'+str(3)+'.csv', delimiter=',')
    
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

    print(eval_labels)
    
    # Reshape images to apply
    #training_images = np.reshape(training_images,(length*widths*channels,10000))
    #eval_images = np.reshape(eval_images,(length*widths,numb_eval))

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
        model_fn=cnn_model_fn, model_dir="/Users/HK/Mathematics/Projects/time_series_pictures/model_data")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_images},
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


# Our application logic will be added here

if __name__ == "__main__":
    tf.app.run()
