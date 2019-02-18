#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
from collections import Counter
import multiprocessing
import time
from joblib import Parallel, delayed

NUM_CORES = multiprocessing.cpu_count()


"""
simple bin method
reference:  https://github.com/artemyk/ibsgd/blob/master
"""


def digitize(lower_bound, upper_bound, num_of_bins, T):
	"""
	cut [-1, 1] into num_of_bins - 1 intervals
	the number in T will be changed to its lower cutoff value
	the value that's <-1 or >= 1 will be changed as 1
	"""
	bins = np.linspace(lower_bound, upper_bound, num_of_bins, dtype='float32')
	T_digitized = bins[np.digitize(np.squeeze(T.reshape(1, -1)), bins) - 1].reshape(T.shape)
	return T_digitized


def get_unique_probs(X):
	"""
	find all the unique x in X to classify X; get p(x)
	"""
	# reshape X to a 2-D matrix
	X = np.reshape(X, (X.shape[0], -1))
	# find IDs to represent every sample in X
	uniqueids = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
	# using IDs to find unique xs and calcu p(x)
	_, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
	#print("unique counts:", len(unique_counts))
	#print("unique_inverse", unique_inverse)
	return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse


# def calc_entropy_for_specific_x(ts_given_x, p_x):
#    """
#    H(T|X==xval)
#    """
#    p_t_given_x, _ = get_unique_probs(ts_given_x)
#    H_T_GIVEN_X = - p_x * np.sum(p_t_given_x * np.log2(p_t_given_x))
#    return H_T_GIVEN_X


def bin_calc_information(X, T, T_lower_bound, T_upper_bound, num_of_bins):
	"""
	MC approach to estimate the MI between X and T, binning the T
	X: X.shape[0] should be the number of test samples; put y as X and get I(Y;T)
	T: the data of hidden layers, corresponding to X. also T.shape[0] is the number of test samples
	num_of_bins: how many intervals [-1, 1] is cut into
	"""
	p_xs, unique_inverse_x = get_unique_probs(X)

	T_digitized = digitize(T_lower_bound, T_upper_bound, num_of_bins, T)
	#print("T_digitized sum unique", np.unique(np.sum(T_digitized, axis=1)))
	p_ts, _ = get_unique_probs(T_digitized)

	H_T = -np.sum(p_ts * np.log2(p_ts))

	H_T_GIVEN_X = 0.0

	# H_T_GIVEN_X = np.array(Parallel(n_jobs=2)(delayed(calc_entropy_for_specific_x)
	 #                               (T_digitized[unique_inverse_x == xval, :], p_xs[xval])
	  #                        for xval in np.unique(unique_inverse_x)))
	# H_T_GIVEN_X = np.sum(H_T_GIVEN_X)

	for xval in np.unique(unique_inverse_x):
		p_t_given_x, _ = get_unique_probs(T_digitized[unique_inverse_x == xval, :])
		H_T_GIVEN_X += - p_xs[xval] * np.sum(p_t_given_x * np.log2(p_t_given_x))
	#print("p_ts", p_ts)
	#print("H_T", H_T)
	#print("H_T_GIVEN_X", H_T_GIVEN_X)
	return H_T - H_T_GIVEN_X


"""
bin method
another approach using binning to calculate MI
can be used to test other approaches
"""


def bin_calc_information_2(X, Y, T, T_lower_bound, T_upper_bound, num_of_bins):
	"""
	every sample in X should be different
	the value of T should be within [-1, 1], which totally depends on digitize()
	"""
	# digitize T
	T = digitize(T_lower_bound, T_upper_bound, num_of_bins, T)
	# reshape to 2-D matrix
	X = X.reshape(X.shape[0], -1)
	T = T.reshape(T.shape[0], -1)
	Y = Y.reshape(Y.shape[0], -1)
	# the number of samples
	N = np.shape(X)[0]
	# the dimension of data
	dims_x = np.shape(X)[1]
	dims_t = np.shape(T)[1]
	dims_y = np.shape(Y)[1]
	# the "translator" vector
	vector_x = np.zeros([dims_x, 1])
	for i in range(dims_x):
		vector_x[i] = 10**(dims_x - i - 1)
	vector_t = np.zeros([dims_t, 1])
	for i in range(dims_t):
		vector_t[i] = 10**(dims_t - i - 1)
	vector_y = np.zeros([dims_y, 1])
	for i in range(dims_y):
		vector_y[i] = 10**(dims_y - i - 1)
	# change every line in matrix into a number eg. [1, 3, 4]->134
	X = np.dot(X, vector_x)
	T = np.dot(T, vector_t)
	Y = np.dot(Y, vector_y)
	# reshape it one-dimension list to fit Counter's requirement
	X = np.reshape(X, [-1])
	T = np.reshape(T, [-1])
	Y = np.reshape(Y, [-1])
	# build dictionary for [t]
	dict_T = Counter(T)
	# build dictionary for [y]
	dict_Y = Counter(Y)
	# build dictionary for [(x, t)]
	X_T = []
	for i in range(N):
		X_T.append((X[i], T[i]))
	dict_XT = Counter(X_T)
	# build dictionary for [(y, t)]
	Y_T = []
	for i in range(N):
		Y_T.append((Y[i], T[i]))
	dict_YT = Counter(Y_T)
	# calculate I_XT
	I_XT = 0
	for key in dict_XT:
		x = key[0]
		t = key[1]
		p_x = float(1)/N
		p_t = float(dict_T[t])/N
		p_xt = float(dict_XT[key])/N
		I_XT += p_xt * np.log2(p_xt / (p_x * p_t))
	# calculate I_YT
	I_YT = 0
	for key in dict_YT:
		y = key[0]
		t = key[1]
		p_y = float(dict_Y[y]) / N
		p_t = float(dict_T[t]) / N
		p_yt = float(dict_YT[key]) / N
		I_YT += p_yt * np.log2(p_yt / (p_y * p_t))
	return I_XT, I_YT

"""
now it's kde method
reference:  https://github.com/artemyk/ibsgd/blob/master
"""


def get_dists(X):
	"""
	compute the pairwise distance matrix for a set of
	vectors specifie by the matrix X.
	"""
	#print("***get_dists start***")
	#t0 = time.time()
	x2 = np.expand_dims(np.sum(np.square(X), axis=1), 1)
	#t1 = time.time()
	#print("1)", t1 - t0)

	dists = x2 + x2.T - 2*np.dot(X, X.T)
	#t2 = time.time()
	#print("2)", t2 - t1)

	#print("***get_dists end***")
	return np.clip(dists, a_min=0, a_max=None)  # Sometimes rounf-off error will generate minus number


def get_shape(x):
	"""
	x should be a 2-D matrix.
	N is the number of samples, dims is the dimension of every sample vector
	"""
	N = float(x.shape[0])
	dims = float(x.shape[1])
	return N, dims


def entropy_estimator_kl(x, var):
	"""
	KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
	the variable is seen as a mixture of gaussian distribution, dist center for every component is the sample value of x
	e.g. if there are N samples in x, then this is a dims-dimension variable that subjects to GMM with N centers
	there shouldn't be repeating samples in x
	"""
	#print("======entropy_estimator_kl start=========")
	#t0 = time.time()
	N, dims = get_shape(x)
	#t1 = time.time()
	#print("1", t1 - t0)

	dists = get_dists(x)  # cost most of the time in calculating MI
	#t2 = time.time()
	#print("2", t2 - t1)


	# print(dists)
	# print(np.clip(np.sum(np.exp(- dists / (2*var)), axis=1), a_min=1e-50, a_max=None))
	lprobs = np.log(np.clip(np.sum(np.exp(- dists / (2*var)), axis=1), a_min=1e-40, a_max=None))
	#t3 = time.time()
	#print("3", t3 - t2)


	h = -np.mean(lprobs)
	#t4 = time.time()
	#print("4", t4 - t3)


	const = dims/2*(1 + np.log(2*np.pi*var)) + np.log(N)
	#t5 = time.time()
	#print("5", t5 - t4)

	#print("======entropy_estimator_kl end=========")
	return const + h


def entropy_estimator_bd(x, var):
	"""
	Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I
	"""
	N, dims = get_shape(x)
	val = entropy_estimator_kl(x, 4*var)
	return val + np.log(0.25)*dims/2


def kde_gaussian_entropy(output, var):
	"""
	Return entropy of a multivariate Gaussian, in nats
	"""
	dims = output.shape[1]
	return (dims/2.0)*(np.log(2*np.pi*var) + 1)


def kde_calc_information(X, Y, T, noise_variance):
	"""
	calculate MI bound for XT, YT
	"""
	# reshape to 2-D matrix
	t0 = time.time()
	X = np.reshape(X, (X.shape[0], -1))
	T = np.reshape(T, (T.shape[0], -1))
	Y = np.reshape(Y, (Y.shape[0], -1))
	_t = time.time() - t0
	print("reshaping data spent %s s" % _t)

	# nats to bits conversion factor
	nats2bits = 1.0 / np.log(2)

	# mutual info between X & T
	t0 = time.time()
	H_T_GIVEN_X = kde_gaussian_entropy(T, noise_variance)
	_t = time.time() - t0
	print("calculating H(T|X) spent %s s" % _t)

	t0 = time.time()
	H_T_upper = entropy_estimator_kl(T, noise_variance)
	H_T_lower = entropy_estimator_bd(T, noise_variance)
	_t = time.time() - t0
	print("calculating H(T) upper & lower bound spent %s s" % _t)

	MI_XT_upper = nats2bits * (H_T_upper - H_T_GIVEN_X)
	MI_XT_lower = nats2bits * (H_T_lower - H_T_GIVEN_X)
	# mutual info between Y & T
	p_ys, unique_inverse_y = get_unique_probs(Y)

	H_T_GIVEN_Y_upper = 0.
	H_T_GIVEN_Y_lower = 0.

	t0 = time.time()
	for yval in np.unique(unique_inverse_y):
		H_T_GIVEN_y_upper = entropy_estimator_kl(T[unique_inverse_y == yval, :], noise_variance)
		H_T_GIVEN_y_lower = entropy_estimator_bd(T[unique_inverse_y == yval, :], noise_variance)
		H_T_GIVEN_Y_upper += p_ys[yval] * H_T_GIVEN_y_upper
		H_T_GIVEN_Y_lower += p_ys[yval] * H_T_GIVEN_y_lower
	_t = time.time() - t0
	print("calculating H(T|Y) upper & lower bound spent %s s" % _t)

	MI_YT_upper = nats2bits * (H_T_upper - H_T_GIVEN_Y_upper)
	MI_YT_lower = nats2bits * (H_T_lower - H_T_GIVEN_Y_lower)
	return MI_XT_upper, MI_XT_lower, MI_YT_upper, MI_YT_lower