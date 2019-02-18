import numpy as np
import scipy.io as sio
from keras.utils import to_categorical


# read the data
print ('Loading Data...')
data = sio.loadmat('var_u.mat')
F = data['F']  # 4096*12
y = data['y'].T  # 4096*1
y = to_categorical(y)  # one-hot 4096*2
print ('Loaded')

# parameters
LAYER_SIZE = [10]
EPOCHS = 8000
BATCH_SIZE = 512
lr = 0.0004
RATE_DECAY = 1


def shuffle(a, b):
	"""Shuffle the arrays randomly"""
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]


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
	a = num / (denm + 1e-8)
	return a

def tanh(z):
	return np.clip(z, 0, None)

# initialize
W_1 = np.random.normal(0., 1.0/np.sqrt(12), size=(12, LAYER_SIZE[0]))
W_2 = np.random.normal(0., 1.0/np.sqrt(LAYER_SIZE[0]), size=(LAYER_SIZE[0], 2))

b_1 = np.zeros(LAYER_SIZE[0])
b_2 = np.zeros(2)
#b_1 = np.random.normal(0., 1.0/np.sqrt(12), size=LAYER_SIZE[0])
#b_2 = np.random.normal(0., 1.0/np.sqrt(LAYER_SIZE[0]), size=2)

C = 0.
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


# randomly shuffle the data
F, y = shuffle(F, y)

# training
for i in range(EPOCHS):
	# report training

	for j in range(int(np.size(y, 0)/BATCH_SIZE)):
		# training step
		step += 1

		#input layer a_0
		a_0 = F[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :]

		a_1 = np.dot(a_0, W_1) + b_1
		A_1 = tanh(a_1)

		# output layer a_6
		a_2 = np.dot(A_1, W_2) + b_2
		# softmax layer
		A_2 = softmax(a_2)

		# gradients
		g_a2 = 1. / BATCH_SIZE *(A_2 - y[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :]) # (BATCH_SIZE, layer_size)

		g_A1 = np.dot(g_a2, W_2.T)
		g_a1 = np.where(a_1 > 0, 1., 0.) * g_A1

		g_W1 = np.dot(a_0.T, g_a1)
		g_W2 = np.dot(a_1.T, g_a2)

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
		A_1 = tanh(a_1)
		a_2 = np.dot(A_1, W_2) + b_2
		A_2 = softmax(a_2)

		# loss
		L = 1. / y.shape[0] * np.sum(-np.log(A_2 + 1e-8) * y)
		print("the loss：%s" % L)
		#print(a_6[900:1000])

		predict = (np.amax(A_2, axis=1, keepdims=True) == A_2)
		predict = np.where(predict, 1., 0.)
		accur = float(np.sum(predict * y)) / y.shape[0]

		#prediction_class = np.round(A_2)
		#accuracy = np.sum(y == prediction_class, axis=1) / y.shape[1]
		#accuracy = float(sum(accuracy)) / F.shape[0]
		print('the accuracy of DNN is: %s' % accur)

print("Training finished")