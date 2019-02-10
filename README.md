# Classifying telemetry data with convolutional neural networks

## Introduction

One area where the power of deep learning has been particularly evident is image recognition and classification. Many of the most impressive applications of neural networks to image analysis have used the convolutional neural network (CNN) architecture which is able to capture the local structure in an image in the initial layers of the network and later combine this information to be able to make global statements. Note that in their standard form, CNN's assume that the data in question has a 2-dimensional structure.

In this project we seek to leverage the power of CNN's to better classify time series data. While time series data does not in general have a 2-dimensional structure, it can be mapped from a length *t* vector (where *t* is the number of recorded time steps) to a size *t x m* array using a mathematical tool called a convolution (here *m* is a positive integer chosen by the user). Depending on the width and size of the convolution, different scales of detail from the time series will be captured. In the end therefore, the "image" corresponding to a time series will not contain any new information, but small-scale vs. large-scale features of the time series will appear as local structure in different parts of the image.

## The convolution map

While there are many possible convolutions that could be used, in this project we have chosen to use the [Ricker wavelet](https://en.wikipedia.org/wiki/Mexican_hat_wavelet) (or Mexican hat wavelet). This is the negative normalized second derivative of the Gaussian function. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\psi(t)&space;=&space;\frac{2}{\sqrt{3}\pi^{1/4}}e^{-\frac{t^2(1-t^2)}{2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\psi(t)&space;=&space;\frac{2}{\sqrt{3}\pi^{1/4}}e^{-\frac{t^2(1-t^2)}{2}}" title="\psi(t) = \frac{2}{\sqrt{3}\pi^{1/4}}e^{-\frac{t^2(1-t^2)}{2}}" /></a>

## The data set

The data set we are working with consists of telemetry data for mice. From each mouse we have a time series with both temperature and movement data from the mice. At a specified point in the time series 

## Weaknesses to the approach

While our approach yields a method whereby structure at different scales in a time series can be fed as features to a CNN, there are some important trade-off's which should be noted. The first is that there is a cost to converting time series into images, in that we move from dealing with objects of size *t* to objects of size *tm*. When *t* is large, then in order to capture structure at all scales we will need to make *m* fairly large as well. This may be prohibitively expensive in some cases. To mitigate this we have: (1) chosen to classify subwindows of each time series of shorter length, (2) 
