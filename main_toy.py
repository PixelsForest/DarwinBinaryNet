import numpy as np
import scipy.io as sio
import information_toolbox as it
import matplotlib.pyplot as plt
from keras.utils import to_categorical
"""One layer binary network with continuous weights"""

# read the data
print ('Loading Data...')
data = sio.loadmat('var_u.mat')
F = data['F']  # 4096*12
y = data['y'].T  # 4096*1
y = to_categorical(y)  # one-hot 4096*2
print ('Loaded')

# parameters
ITERATION = 50
PLOT = True
LAYER_SIZE = [28]
EPOCHS = 20000
BATCH_SIZE = 512
LR = 0.00002
RATE_DECAY = 1

# plot data
axis_1 = []
axis_2 = []
epoch_index = []
accuracy = 0.

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


def binarize(x):
	# binarize x to 0 1 by comparing it with 0.5
	x = x + 0.5
	bi = np.clip(x, 0., 1.999)
	bi = (bi.astype(np.int))
	return bi.astype(np.float)


def softmax(z):
	# input: N * layer size ; output:N * layer size
	num = np.exp(z)
	denm = np.sum(np.exp(z), axis=1, keepdims=True)
	a = num / denm
	return a


def Adam(m, v, gradient, t, b1=0.9, b2=0.999):
	# m, v are lists but only store one element
	# m[0] = ... actually changes the global variable(e.g. m_gamma)
	if m:
		m[0] = b1 * m[0] + (1 - b1) * gradient
		v[0] = b2 * v[0] + (1 - b2) * (gradient * gradient)
	else:
		m[0] = gradient
		v[0] = gradient * gradient
	m_hat = m[0] / (1 - np.power(b1, t))
	v_hat = v[0] / (1 - np.power(b2, t))
	return m_hat / (np.sqrt(v_hat) + 1e-9)


# training
for t in range(ITERATION):
	print("********************* iteration %s ***********************" % t)
	# plotting
	if PLOT and axis_1:
		fig = plt.figure()
		# plot
		font = {'family': 'serif',
				'color': 'black',
				'weight': 'normal',
				'size': 16}
		cmap = plt.cm.get_cmap('gnuplot')

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
		plt.title('IB_BIN %s' %accuracy)
		plt.savefig("data/main_toy/IB_BIN_ITERATION%s.png" % (t-1))
		axis_1 = np.array(axis_1)
		axis_2 = np.array(axis_2)
		epoch_index = np.array(epoch_index)
		np.save('data/main_toy/IB_IXT_BIN_ITERATION%s.npy' % (t-1), axis_1)
		np.save('data/main_toy/IB_IYT_BIN_ITERATION%s.npy' % (t-1), axis_2)
		np.save('data/main_toy/IB_epoch_color_BIN', epoch_index)

		# reinitialize
		axis_1 = []
		axis_2 = []
		epoch_index = []

	# initialize
	W_1 = np.random.normal(0., 5.0 / np.sqrt(12), size=(12, LAYER_SIZE[0]))
	W_2 = np.random.normal(0., 5.0 / np.sqrt(LAYER_SIZE[0]), size=(LAYER_SIZE[0], 2))

	b_1 = np.random.normal(0., 5.0 / np.sqrt(12), LAYER_SIZE[0])
	b_2 = np.random.normal(0., 5.0 / np.sqrt(LAYER_SIZE[0]), 2)

	C = 0.
	lr = LR
	# Adam optimizer
	step = 0

	m_W1 = [0.]
	m_W2 = [0.]
	v_W1 = [0.]
	v_W2 = [0.]

	m_b1 = [0.]
	m_b2 = [0.]
	v_b1 = [0.]
	v_b2 = [0.]

	for i in range(EPOCHS):
		# calc the data for information plane
		if PLOT and epoch_ctrl(i) and i:
			a_0 = F
			a_1 = np.dot(a_0, W_1) + b_1
			a_1b = binarize(a_1)
			a_2 = np.dot(a_1b, W_2) + b_2
			A_2 = softmax(a_2)

			I_XT1_BIN = it.bin_calc_information(F, a_1b, 0, 1, 2)  # softmax belongs to 0~1
			I_YT1_BIN = it.bin_calc_information(y, a_1b, 0, 1, 2)
			# store MI
			axis_1.append([I_XT1_BIN])
			axis_2.append([I_YT1_BIN])
			epoch_index.append(float(i) / EPOCHS)

		# training
		for j in range(int(np.size(y, 0)/BATCH_SIZE)):
			# training step
			step += 1

			#input layer a_0
			a_0 = F[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :]

			a_1 = np.dot(a_0, W_1) + b_1
			a_1b = binarize(a_1)

			# output layer a_6
			a_2 = np.dot(a_1b, W_2) + b_2
			# softmax layer
			A_2 = softmax(a_2)

			# gradients
			g_a2 = 1. / BATCH_SIZE *(A_2 - y[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :]) # (BATCH_SIZE, layer_size)

			g_a1b = np.dot(g_a2, W_2.T)
			g_a1 = g_a1b

			g_W1 = np.dot(a_0.T, g_a1)
			g_W2 = np.dot(a_1b.T, g_a2)

			g_b1 = np.sum(g_a1, axis=0)
			g_b2 = np.sum(g_a2, axis=0)

			#update
			W_1 = W_1 - lr * Adam(m_W1, v_W1, g_W1, step)
			W_2 = W_2 - lr * Adam(m_W2, v_W2, g_W2, step)

			b_1 = b_1 - lr * Adam(m_b1, v_b1, g_b1, step)
			b_2 = b_2 - lr * Adam(m_b2, v_b2, g_b2, step)

			# learning rate decay
			lr = lr * RATE_DECAY


		# testing it
		if (i-1) % 100 == 0: # i % 100 == 0 and i > 0
			print("epoch : %s " % i)
			a_0 = F
			a_1 = np.dot(a_0, W_1) + b_1
			a_1b = binarize(a_1)
			a_2 = np.dot(a_1b, W_2) + b_2
			A_2 = softmax(a_2)

			# loss
			L = 1. / y.shape[0] * np.sum(-np.log(A_2 + 1e-8) * y)
			print("the loss：%s" % L)
			#print(a_6[900:1000])

			predict = (np.amax(A_2, axis=1, keepdims=True) == A_2)
			predict = np.where(predict, 1., 0.)
			accur = float(np.sum(predict * y)) / y.shape[0]
			print('the accuracy of DNN is: %s' % accur)
			accuracy = accur






print("Training finished")