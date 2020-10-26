# Traffic-Sign-Classifier
This repo is for the fullfilment of the Udacity CarND Nanodegree
## Overview

The goal of this repo is to demonstrate a Machine Learning approach to detect and deduce the type of Traffic Sign Classifier for Autonomous Vehicle. The ML archietecture used in this project is a LeNet Archietecture. The repo was programmed using the numpy and tensorflow packages of Python3.

## Purpose

In every day driving, the trafiic signs helps the commuter to identify the various warnings and rules. Identifying these signs is a necessasity to make a Vehicle autonomous.

## LeNet

The Machine learning Archietecture used to train a model with the features and weights is the LeNet Archietecture. The final archetecture implemented is given by: 

 Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image   							| 
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 
| RELU				|
| Convolution 3x3	| 1x1 stride,Valid padding,outputs 10x10x16 
| RELU     		|
| Max Polling		| 2x2 stride,Valid Padding, outputs 5x5x16
| Flatten
| Fully connected		| Input = 400 Output = 120
| RELU
| Fully Connected		| Input = 120 Output = 84
| RELU
| Fully Connected		| Input = 84 Output = 44
| Dropout 
| Softmax			        									|
|						|												|
|						|												|
 

## Project Requirements and Dependencies

Python3

OpenCv

Tensorflow

Numpy

## Command to Run

```
python3 classifier.py

```

## Licence
The Repository is Licensed under the MIT License.
```
MIT License

Copyright (c) 2019 Charan Karthikeyan Parthasarathy Vasanthi, Nagireddi Jagadesh Nischal, Sai Manish V

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
