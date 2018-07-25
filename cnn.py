import cPickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
# import os
import pandas as pd

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def Compute(df):
	images = df[1]

	l = []

	for name in images:
		file = tf.read_file('../dataset/VQAMed2018Train/VQAMed2018Train-images/'+name+'jpg')
		img = tf.image.decode_jpeg(file, channels=3)
		resized_image = tf.image.resize_images(img, [300, 300])
		print  name+" "+str(resized_image)
		l.append(resized_image)
	

	X = tf.placeholder(tf.float32, shape=(None,300,300, 3), name="X")

	with tf.name_scope("cnn"):
		conv1 = tf.layers.conv2d(inputs=X,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu,name='conv1')
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2,name='pool1')
		norm1 = tf.layers.batch_normalization(inputs=pool1,axis=-1,name = "norm1")
		conv2 = tf.layers.conv2d(inputs=norm1,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu,name='conv2')
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],strides=2,name="pool2")
		norm2 = tf.layers.batch_normalization(inputs=pool2,axis=-1,name = "norm2")
		norm2 = tf.contrib.layers.flatten(norm2)
		fc1 = tf.layers.dense(inputs=norm2, units=256, activation=tf.nn.relu,name="fc1")
		fc2 = tf.layers.dense(inputs=fc1, units=128, activation=tf.nn.relu,name="fc2")

if __name__ == '__main__':
	#Reading File
	df = pd.read_csv('../dataset/VQAMed2018Train/VQAMed2018Train-QA.csv', sep='\t',header=None)

	# Compute(df)
	for quesition in df[2]:
		print length 
	