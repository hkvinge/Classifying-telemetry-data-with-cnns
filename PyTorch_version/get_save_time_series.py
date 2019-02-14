from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
sys.path.append('/data3/darpa/calcom/')
import calcom
import numpy as np
import matplotlib.pyplot as plt
import time
from random import randint
import utils

# Set length of time interval to investigate
length = 2400
widths = 50
channels = 1
step_size = 50

# Path to data files
path_to_data = "/data3/darpa/tamu/"

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

# Save the numpy arrays to csv files
np.savetxt("label_train.csv", labels_train, delimiter=",")
np.savetxt("label_test.csv", labels_test, delimiter=",")
np.savetxt("temps_train.csv", temps_train, delimiter=",")
np.savetxt("temps_test.csv", temps_test, delimiter=",")
