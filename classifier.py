"""
@file : classifier.py
@author : Charan Karthikeyan P V
@License : MIT License
@date :08/24/2020
@brief : This file imports the data, trains it and outputs a model for Traffic
sign classification
"""

import pickle 
import numpy as np
import cv2
import pickle 
import csv
import matplotlib.pyplot as plt 
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

"""
@brief : Function to get the values from the pickle file
@param : None
@return : The train, valid and test values with the checkpoint
"""
def read_data():
	training_file = "data/train.p"
	validation_file= "data/valid.p"
	testing_file = "data/test.p"

	with open(training_file, mode='rb') as f:
		train = pickle.load(f)
	with open(validation_file, mode='rb') as f:
		valid = pickle.load(f)
	with open(testing_file, mode='rb') as f:
		test = pickle.load(f)
		
	X_train, y_train = train['features'], train['labels']
	X_valid, y_valid = valid['features'], valid['labels']
	X_test, y_test = test['features'], test['labels']

	checkpoint1_var = {}
	checkpoint1_file = "output/checkpoint1"
	# Uncomment this to see the number of classes in the 
	# training, validation and testing files
	# print("Training Set: {}".format(len(training_file)))
	# print("Validation Set: {}".format(len(validation_file)))
	# print("Testing Set: {}".format(len(testing_file)))

	return X_train, y_train, X_valid, y_valid, X_test, y_valid, checkpoint1_var

"""
@brief : Function to print out he details of the data 
@param : The images and labels of the test, validation and test files
@return : The number of classes in the dataset 
"""
def general_summary(X_train, y_train, X_valid, y_valid, X_test, y_valid):
	n_train = len(X_train)
	n_test = len(X_test)
	n_valid = len(X_valid)

	# The shape of the traffic sign image 
	img = random.randint(0, len(X_train))
	image_shape = X_train[img].shape

	# The number of unique classes in the dataset
	n_classes = np.loadtxt('signnames.csv', dtype='str', delimiter=',',
	 usecols=(0, 1), unpack=True).shape[1]
	n_class = len(set(y_train))

	# Uncomment this code to get the print the number of training, testing
	# and validation samples

	# print("Number of training examples =", n_train)
	# print("Number of testing examples =", n_test)
	# print("Image data shape =", image_shape)
	# print("Number of classes =", n_classes)

	return n_classes

"""
@brief : Function to visualize the data with their labels.
@param : X_train -> The training images.
		 y_train -> The labels of the training images.
		 n_class -> The number of classes in the file.
		 checkpoint1_var -> The empty list to get the labels list. 
@return : None
"""
def data_visualization(X_train, y_train, n_class, checkpoint1_var):
	label_dict = None
	fig = plt.figure(figsize(32, 32), tight_layout = {'h_pad':10})
	with open("signnames.csv", mode = 'r') as file:
		reader = csv.reader(file)
		next(reader, None)
		label_dict = {int(rows[0]):rows[1] for rows in reader}
	checkpoint1_var["label_dict"]  = label_dict
	# Loads the images and shows them as a single image with their labels
	for i in range(n_class):
		pos = np.where(y_train == i)
		img = X_train[pos[0][0]]
		ax = fig.add_subplot(int(n_class/4)+1, 4, i+1)
		ax.imshow(img,interpolation = 'none')
		ax.set_title(label_dict[y_train[pos[0][0]]])
	plt.show()
	# Number of data per class visualization
	y_train1 = pd.DataFrame()
	y_train1['label'] = y_train
	ax = y_train1['label'],value_counts().plot(kind = 'barh', figsize =(5, 50),
		title = 'Samples per class')
	ax.set_yticklabels(list(map(lambda x: label_dict[x],
	 y_train1['label'].value_counts().index.tolist())))            
	for i, v in enumerate(y_train1['label'].value_counts()):
		labels = ax.text(v + 10, i - 0.25, str(v), color='red')

"""
@brief : Function to preprocess the images before sending to the network
@param : image -> The Image to be converted
@return : The converted image
"""
def preprocess(image):
	images = np.ndarray((image.shape[0], 32, 32, 1), dtype= "uint8")
	for i, img in enumerate(image):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.equalizeHist(img)
		img = np.expand_dims(img, axis=2)
		img = img.astype('float32')
		images[i] = img
	return images

"""
@brief : Function to declare the LeNet archietecture for the classification model 
@param : x-> The input images
@return : The final logits after the training module
"""
def LeNet(x):
	mu = 0
	sigma = 0.001
	dropout = 1
	conv1_W = tf.Variable(tf.truncated_normal(shape=(5,5,1,6),mean = mu, stddev = sigma ))
	conv1_b = tf.Variable(tf.zeros(6))
	conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

	# SOLUTION: Activation.
	conv1 = tf.nn.relu(conv1)

	# SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
	conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	# SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
	conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
	conv2_b = tf.Variable(tf.zeros(16))
	conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
	
	# SOLUTION: Activation.
	conv2 = tf.nn.relu(conv2)

	# SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
	conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	# SOLUTION: Flatten. Input = 5x5x16. Output = 400.
	fc0   = flatten(conv2)
	
	# SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
	fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
	fc1_b = tf.Variable(tf.zeros(120))
	fc1   = tf.matmul(fc0, fc1_W) + fc1_b
	
	# SOLUTION: Activation.
	fc1    = tf.nn.relu(fc1)

	# SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
	fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
	fc2_b  = tf.Variable(tf.zeros(84))
	fc2    = tf.matmul(fc1, fc2_W) + fc2_b
	
	# SOLUTION: Activation.
	fc2    = tf.nn.relu(fc2)
	

	# SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 44.
	fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 44), mean = mu, stddev = sigma))
	fc3_b  = tf.Variable(tf.zeros(44))
	fc2 = tf.nn.dropout(fc2, dropout)
	
	
	logits = tf.matmul(fc2, fc3_W) + fc3_b
	
	return logits 
 """
 @brief : Fuction to calculate the accuaracy of the generated model
 @param : X_data -> The images from the data to get the accracy on 
		  y_data -> The labels of the image from the data file
 @return : Accuracy
 """
def evaluate(X_data, y_data):
	num_examples = len(X_data)
	total_accuracy = 0
	sess = tf.get_default_session()
	for offset in range(0, num_examples, BATCH_SIZE):
		batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
		accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
		#loss = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y})
		total_accuracy += (accuracy * len(batch_x))
	return total_accuracy / num_examples

"""
@brief : The main function to execute the process in an ordered fashion 
@param : None
@return : None
"""
def main():
	X_train, y_train, X_valid, y_valid, X_test, y_valid, checkpoint1_var = read_data()
	n_classes = general_summary(X_train, y_train, X_valid, y_valid, X_test, y_valid)
	
	# Uncomment the next line if you want to see the data visualization
	# data_visualization(X_train, y_train, n_class, checkpoint1_var)	
	EPOCHS = 100
	BATCH_SIZE = 200
	X_train, y_train = shuffle(X_train,y_train)
	X_validation, y_validation = X_valid, y_valid

	#Convert the images to the required size 
	X_train = preprocess(X_train)
	X_validation = preprocess(X_validation)
	X_test = preprocess(X_test)

	# Network array definitions
	x = tf.placeholder(tf.float32, (None, 32,32,1))
	y = tf.placeholder(tf.int32, (None))
	keep_prob = tf.placeholder(tf.float32)
	one_hot_y = tf.one_hot(y, 44)
	rate = 0.001
	logits = LeNet(x)
	y_pred = tf.nn.softmax(logits)
	pred = tf.argmax(y_pred, dimension=1)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
	loss_operation = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate =rate)
	training_operation = optimizer.minimize(loss_operation)

	#prediction steps
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
	accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	# Model Saver
	saver = tf.train.Saver()



