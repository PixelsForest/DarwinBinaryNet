#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import information_toolbox as it
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""define some useful functions"""


def add_layer(inputs, in_size, out_size, activation_function=None):
	"""
	 inputs: input data matrix, one sample in a line
	 in_size: length of the last layer
	 out_size: length of this layer
	 activation_function: sigmoid
	 return: values in this layer
	"""
	Weights = tf.Variable(tf.truncated_normal([in_size, out_size],
											  stddev=1/tf.sqrt(tf.cast(in_size, dtype=tf.float32)))) # tf.random_normal([in_size, out_size])
	bias = tf.Variable(tf.zeros([1, out_size]))  # tf.random_uniform([1, out_size], minval=-0.0001, maxval=0.0001) # tf.zeros([1, out_size])
	Wx_plus_b = tf.matmul(inputs, Weights)+bias
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs


def shuffle(a, b):
	"""Shuffle the arrays randomly"""
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]


def epoch_ctrl(epoch):
	"""give the right epoch to compute MI"""
	if epoch < 20:  # Log for all first 20 epochs
		return True
	elif epoch < 100:  # Then for every 5th epoch
		return (epoch % 5 == 0)
	elif epoch < 2000:  # Then every 10th
		return (epoch % 20 == 0)
	else:  # Then every 100th
		return (epoch % 100 == 0)

"""network"""

cfg = {}
# iterate the experiment for several times
cfg['ITERATION'] = 1
# network configuration
cfg['SGD_LEARNINGRATE'] = 0.0004
cfg['NUM_EPOCHS'] = 8000
cfg['ACTIVATION'] = 'tanh'
# cfg['ACTIVATION'] = 'relu'
# cfg['ACTIVATION'] = 'softsign'
# cfg['ACTIVATION'] = 'softplus'
# How many hidden neurons to put into each of the layers
cfg['LAYER_DIMS'] = [10]  # original IB network
cfg['NUM_BATCHS'] = 8
# kde/bin method
cfg['KDE'] = False
cfg['BIN'] = False
cfg['NUM_BINS'] = 30
cfg['VARIANCE'] = 0.001
# epochs to plot MI
cfg['DO_EPOCH_PLOT'] = np.unique(np.logspace(np.log2(1.), np.log2(float(cfg['NUM_EPOCHS'])), 400, dtype=int, base=2)) - 1


print ('Loading Data...')
data = sio.loadmat('var_u.mat')
F = data['F']  # 4096*12
y = data['y'].T  # 4096*1
y = to_categorical(y)  # one-hot 4096*2
print ('Loaded')

with tf.device('/cpu:0'):
	# input data and labels
	input_layer = tf.placeholder(tf.float32, [None, F.shape[1]])
	labels = tf.placeholder(tf.float32, [None, y.shape[1]])

	# hidden layers
	l1 = add_layer(input_layer, F.shape[1], cfg['LAYER_DIMS'][0], activation_function=tf.nn.tanh)
	#l2 = add_layer(l1, cfg['LAYER_DIMS'][0], cfg['LAYER_DIMS'][1], activation_function=tf.nn.tanh)
	#l3 = add_layer(l2, cfg['LAYER_DIMS'][1], cfg['LAYER_DIMS'][2], activation_function=tf.nn.tanh)
	#l4 = add_layer(l3, cfg['LAYER_DIMS'][2], cfg['LAYER_DIMS'][3], activation_function=tf.nn.tanh)
	#l5 = add_layer(l4, cfg['LAYER_DIMS'][3], cfg['LAYER_DIMS'][4], activation_function=tf.nn.tanh)
	# output layer
	output_layer = add_layer(l1, cfg['LAYER_DIMS'][-1], y.shape[1], activation_function=tf.nn.softmax)
	# loss function
	cross_entropy = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(output_layer, 1e-10, 1.0)), axis=1) #  output_layer + 1e-10
	loss = tf.reduce_mean(cross_entropy)

	train_step = tf.train.AdamOptimizer(cfg['SGD_LEARNINGRATE']).minimize(loss)

	# Example code for calling functions from information_toolbox.py
	'''
	import information_toolbox as it
	l1_digitize = it.digitize(num_bins, l1)
	bin_calc_information(X, T, num_of_bins)
	'''

for iteration in range(cfg['ITERATION']):
	# training and print loss
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	# lists to store plot data
	axis_1 = []
	axis_2 = []
	axis_3 = []
	axis_4 = []
	# remember epoch index
	epoch_index = []
	# SGD
	for epoch in range(cfg['NUM_EPOCHS']):
		# randomly shuffle the data
		F, y = shuffle(F, y)
		# pass through every sample in one training epoch
		for batch in range(cfg['NUM_BATCHS']):
			batch_size = F.shape[0] / cfg['NUM_BATCHS']
			sess.run(train_step, feed_dict={input_layer: F[int(batch * batch_size):int((batch + 1) * batch_size), :],
											labels: y[int(batch * batch_size):int((batch + 1) * batch_size), :]})
		if epoch_ctrl(epoch):  # epoch in cfg['DO_EPOCH_PLOT']
			# calculate mutual information
			epoch_index.append(float(epoch) / cfg['NUM_EPOCHS'])
			# report the training progress
			los = sess.run(loss, feed_dict={input_layer:F, labels:y})
			print("iter %s, epoch%s, the loss now is: %s" % (iteration, epoch, los))

			if cfg['BIN']:
				for j in range(1, len(cfg['LAYER_DIMS']) + 1, 1):
					exec ("T%s = sess.run(l%s, feed_dict={input_layer: F[0:128]})" % (j, j))
					exec ("I_XT%s_BIN = it.bin_calc_information(F[0:128], T%s, -1, 1, cfg['NUM_BINS'])" % (j, j))
					exec ("I_YT%s_BIN = it.bin_calc_information(y[0:128], T%s, -1, 1, cfg['NUM_BINS'])" % (j, j))
				T2 = sess.run(output_layer, feed_dict={input_layer: F[0:128]})
				I_XT2_BIN = it.bin_calc_information(F[0:128], T2, 0, 1, 30) # softmax belongs to 0~1
				I_YT2_BIN = it.bin_calc_information(y[0:128], T2, 0, 1, 30)
				# store MI
				axis_1.append([I_XT1_BIN, I_XT2_BIN])
				axis_2.append([I_YT1_BIN, I_YT2_BIN])
			if cfg['KDE']:
				for j in range(1, len(cfg['LAYER_DIMS']) + 1, 1):
					exec ("T%s = sess.run(l%s, feed_dict={input_layer: F})" % (j, j))
					exec ("xu%s, xl%s, yu%s, yl%s = it.kde_calc_information(F, y, T%s, cfg['VARIANCE'])" % (j, j, j, j, j))
				# store MI
				axis_3.append([(xu1 + xl1) / 2, (xu2 + xl2) / 2, (xu3 + xl3) / 2, (xu4 + xl4) / 2, (xu5 + xl5) / 2])
				axis_4.append([(yu1 + yl1) / 2, (yu2 + yl2) / 2, (yu3 + yl3) / 2, (yu4 + yl4) / 2, (yu5 + yl5) / 2])

	# report the final accuracy
	prediction_class = np.round(sess.run(output_layer, feed_dict={input_layer: F}))
	accuracy = np.sum(y == prediction_class, axis=1) / y.shape[1]
	accuracy = float(sum(accuracy)) / F.shape[0]
	print('the final accuracy of DNN is: %s' % accuracy)

	# plot
	font = {'family': 'serif',
			'color': 'black',
			'weight': 'normal',
			'size': 16}
	cmap = plt.cm.get_cmap('gnuplot')

	if cfg['BIN']:
		fig = plt.figure()
		for i in range(len(axis_1)):
			plt.plot(axis_1[i], axis_2[i], marker='o', linestyle='-', markersize=12,
					 markeredgewidth=0.01, linewidth=0.2, color=cmap(epoch_index[i]))

		ax = plt.gca()
		ax.set_xticks(np.linspace(1, 13, 13))
		ax.set_yticks(np.linspace(0, 1, 6))
		# size of xytick font
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)

		ax.set_xlabel('I(X;T)', fontdict=font)
		ax.set_ylabel('I(Y;T)', fontdict=font)
		plt.title('IB_BIN')
		plt.savefig("data/IB_BIN_ITERATION%s.png" % (iteration))
		# plt.show()
	if cfg['KDE']:
		fig = plt.figure()
		for i in range(len(axis_3)):
			plt.plot(axis_3[i], axis_4[i], marker='o', linestyle='-', markersize=12,
					 markeredgewidth=0.01, linewidth=0.2, color=cmap(epoch_index[i]))

		ax = plt.gca()
		ax.set_xticks(np.linspace(1, 13, 13))
		ax.set_yticks(np.linspace(0, 1, 6))
		# size of xytick font
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)

		ax.set_xlabel('I(X;T)', fontdict=font)
		ax.set_ylabel('I(Y;T)', fontdict=font)
		plt.title('IB_KDE, epochs:%s' % cfg['NUM_EPOCHS'])
		plt.savefig("data/IB_KDE_ITERATION%s(loss:%s).png" % (iteration, sess.run(loss, feed_dict={input_layer: F, labels: y})))
		# plt.show()

	# save the data
	if cfg['BIN']:
		axis_1 = np.array(axis_1)
		axis_2 = np.array(axis_2)
		epoch_index = np.array(epoch_index)
		np.save('data/IB_IXT_BIN_ITERATION%s.npy' % iteration, axis_1)
		np.save('data/IB_IYT_BIN_ITERATION%s.npy' % iteration, axis_2)
		np.save('data/IB_epoch_color_BIN', epoch_index)
	if cfg['KDE']:
		axis_3 = np.array(axis_3)
		axis_4 = np.array(axis_4)
		epoch_index = np.array(epoch_index)
		np.save('IB_IXT_KDE_ITERATION%s.npy' % iteration, axis_3)
		np.save('IB_IYT_KDE_ITERATION%s.npy' % iteration, axis_4)
		np.save('IB_epoch_color_KDE', epoch_index)
