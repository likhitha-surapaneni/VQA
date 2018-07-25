import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import urllib
import sys
import os
import zipfile
import tarfile
import json 
import hashlib
import re
import itertools
import cPickle
import time
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def fill_unk(unk):
	global glove_wordmap
	glove_wordmap[unk] = RS.multivariate_normal(m,np.diag(v))
	return glove_wordmap[unk]

def sentence2sequence(sentence):
	tokens = sentence.strip('"(),-').lower().split(" ")
	rows = []
	words = []
	for token in tokens:
		i = len(token)
		while len(token) > 0:
			word = token[:i]
			if word in glove_wordmap:
				rows.append(glove_wordmap[word])
				words.append(word)
				token = token[i:]
				continue
			else:
				i = i-1
			if i==0:
				rows.append(fill_unk(token))
				words.append(token)
				break
		return np.array(rows),words

def contextualize(questions):
	ques= []
	ans=[]
	for q in questions:
		ques.append(sentence2sequence(q))
	return ques

def Compute(df):
	images = df[1]

	l = []

	for name in images:
		file = tf.read_file('VQAMed2018Train-QA.csv'+name+'jpg')
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
	df = pd.read_csv('VQAMed2018Train-QA.csv', sep='\t',header=None)

	glove_wordmap = {}
	glove_vectors_file="glove.6B.50d.txt"
	with open(glove_vectors_file, "r") as glove:
		for line in glove:
			name, vector = tuple(line.split(" ", 1))
			glove_wordmap[name] = np.fromstring(vector, sep=" ")
	wvecs=[]
	for item in glove_wordmap.items():
		wvecs.append(item[1])
	s=np.vstack(wvecs)


	v=np.var(s,0)
	m=np.mean(s,0)
	RS = np.random.RandomState()


	train_data=contextualize(df[2])
	# test_data=contextualize("VQAMed2018Valid-QA.csv")
	print train_data[0][0].shape
