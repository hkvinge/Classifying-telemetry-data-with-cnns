# Classifying telemetry data with convolutional neural networks

## Introduction

One area where the power of deep learning have been particularly evident is image recognition and classification. Many of the most impressive applications of neural networks to image analysis have used the convolutional neural network (CNN) architecture which is able to capture the local structure in an image in the initial layers of the network and later combine this information to be able to make global statements. Note that in their standard form, CNN's assume that the data in question has a 2-dimensional structure.

In this project we seek to leverage the power of CNN's to better classify time series data. While time series data is not in general 2-dimensional, it can be mapped from a length *t* vector (where *t* is the number of time steps) to a size *t x t* array using a mathematical tool called a convolution. Depending on the shape of the convolution, different scales of detail from the time series will be captured. In the end therefore, the "image" corresponding to a time series will not contain any new information, but small-scale vs. large-scale features of the time series will appear as local structure in different parts of the image.

## The convolution map

## The data set

The data set we are working with is a collection of telemetry data for mice. Each mouse 
