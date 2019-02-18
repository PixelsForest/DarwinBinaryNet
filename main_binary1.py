import numpy as np
import scipy.io as sio
from keras.utils import to_categorical


# read the data
print ('Loading Data...')
data = sio.loadmat('var_u.mat')
F = data['F']  # 4096*12
y = data['y'].T  # 4096*1
y = to_categorical(y)  # one-hot 4096*2
# 1 0 to 1 -1
F = 2*F - 1
y = 2*y - 1
print ('Loaded')

# parameters
LAYER_SIZE = [10, 7, 5, 4, 3]
EPOCHS = 8000
BATCH_SIZE = 512
lr = 0.001
RATE_DECAY = 1


def binarize(x):
	bi = np.clip(x, -1., 0.999)
	bi = bi + 1.
	bi = (bi.astype(np.int))*2 - 1
	return bi.astype(np.float)


def STD(x):
	# I_|x|=<1
	x = np.clip(x, -1, 1)
	return np.where(np.abs(x)==1, 0., 1.)


# initialize
W_1 = np.random.normal(0., 1.0/np.sqrt(12), size=(12, LAYER_SIZE[0]))
W_2 = np.random.normal(0., 1.0/np.sqrt(LAYER_SIZE[0]), size=(LAYER_SIZE[0], LAYER_SIZE[1]))
W_3 = np.random.normal(0., 1.0/np.sqrt(LAYER_SIZE[1]), size=(LAYER_SIZE[1], LAYER_SIZE[2]))
W_4 = np.random.normal(0., 1.0/np.sqrt(LAYER_SIZE[2]), size=(LAYER_SIZE[2], LAYER_SIZE[3]))
W_5 = np.random.normal(0., 1.0/np.sqrt(LAYER_SIZE[3]), size=(LAYER_SIZE[3], LAYER_SIZE[4]))
W_6 = np.random.normal(0., 1.0/np.sqrt(LAYER_SIZE[4]), size=(LAYER_SIZE[4], 2))

W_1b = binarize(W_1)
W_2b = binarize(W_2)
W_3b = binarize(W_3)
W_4b = binarize(W_4)
W_5b = binarize(W_5)
W_6b = binarize(W_6)

b_1 = np.zeros(LAYER_SIZE[0])
b_2 = np.zeros(LAYER_SIZE[1])
b_3 = np.zeros(LAYER_SIZE[2])
b_4 = np.zeros(LAYER_SIZE[3])
b_5 = np.zeros(LAYER_SIZE[4])
b_6 = np.zeros(2)

b_1b = binarize(b_1)
b_2b = binarize(b_2)
b_3b = binarize(b_3)
b_4b = binarize(b_4)
b_5b = binarize(b_5)
b_6b = binarize(b_6)

C = 0.

# training
for i in range(EPOCHS):
	# report training
	if i % 100 == 0 and i > 0:
		print("EPOCH: %s, learning rate: %s" %(i, lr))
		print("W_6:", W_6)
		print("g_W6", g_W6)

	# testing it
	if i % 100 == 0 and i > 0:
		a_0 = F
		a_1b = binarize(np.dot(a_0, W_1b) + b_1b)
		a_2b = binarize(np.dot(a_1b, W_2b) + b_2b)
		a_3b = binarize(np.dot(a_2b, W_3b) + b_3b)
		a_4b = binarize(np.dot(a_3b, W_4b) + b_4b)
		a_5b = binarize(np.dot(a_4b, W_5b) + b_5b)
		a_6 = np.dot(a_5b, W_6b) + b_6b
		a_6b = binarize(a_6)
		# loss
		c = y - a_6b
		C = 1. / y.shape[0] * np.sum(np.multiply(c, c))
		print("the loss：%s" %C)
		print(a_6[900:1000])

		accur = np.sum(y == a_6b, axis=1)
		# print("accur:", accur[100:200])
		accur = np.where(accur == 2, 1., 0.)
		accur = float(np.sum(accur)) / y.shape[0]
		print('the accuracy of DNN is: %s' % accur)

	for j in range(int(np.size(y, 0)/BATCH_SIZE)):
		#input layer a_0
		a_0 = F[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :]
		a_1 = np.dot(a_0, W_1b) + b_1b
		a_1b = binarize(a_1)
		a_2 = np.dot(a_1b, W_2b) + b_2b
		a_2b = binarize(a_2)
		a_3 = np.dot(a_2b, W_3b) + b_3b
		a_3b = binarize(a_3)
		a_4 = np.dot(a_3b, W_4b) + b_4b
		a_4b = binarize(a_4)
		a_5 = np.dot(a_4b, W_5b) + b_5b
		a_5b = binarize(a_5)
		# output layer a_6
		a_6 = np.dot(a_5b, W_6b) + b_6b
		a_6b = binarize(a_6)
		# gradients
		g_a6b = 2 * (a_6b - y[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :]) # (BATCH_SIZE, layer_size)
		g_a6 = g_a6b * STD(a_6)
		g_a5b = np.dot(g_a6, W_6b.T)
		g_a5 = g_a5b * STD(a_5)
		g_a4b = np.dot(g_a5, W_5b.T)
		g_a4 = g_a4b * STD(a_4)
		g_a3b = np.dot(g_a4, W_4b.T)
		g_a3 = g_a3b * STD(a_3)
		g_a2b = np.dot(g_a3, W_3b.T)
		g_a2 = g_a2b * STD(a_2)
		g_a1b = np.dot(g_a2, W_2b.T)
		g_a1 = g_a1b * STD(a_1)

		g_W1 = 1. / BATCH_SIZE * (np.dot(a_0.T, g_a1))
		g_W2 = 1. / BATCH_SIZE * (np.dot(a_1b.T, g_a2))
		g_W3 = 1. / BATCH_SIZE * (np.dot(a_2b.T, g_a3))
		g_W4 = 1. / BATCH_SIZE * (np.dot(a_3b.T, g_a4))
		g_W5 = 1. / BATCH_SIZE * (np.dot(a_4b.T, g_a5))
		g_W6 = 1. / BATCH_SIZE * (np.dot(a_5b.T, g_a6))

		g_b1 = 1. / BATCH_SIZE * np.sum(g_a1, axis=0)
		g_b2 = 1. / BATCH_SIZE * np.sum(g_a2, axis=0)
		g_b3 = 1. / BATCH_SIZE * np.sum(g_a3, axis=0)
		g_b4 = 1. / BATCH_SIZE * np.sum(g_a4, axis=0)
		g_b5 = 1. / BATCH_SIZE * np.sum(g_a5, axis=0)
		g_b6 = 1. / BATCH_SIZE * np.sum(g_a6, axis=0)

		#update
		W_1 = np.clip(W_1 - lr * g_W1, -1, 1)
		W_2 = np.clip(W_2 - lr * g_W2, -1, 1)
		W_3 = np.clip(W_3 - lr * g_W3, -1, 1)
		W_4 = np.clip(W_4 - lr * g_W4, -1, 1)
		W_5 = np.clip(W_5 - lr * g_W5, -1, 1)
		W_6 = np.clip(W_6 - lr * g_W6, -1, 1)

		b_1 = np.clip(b_1 - lr * g_b1, -1, 1)
		b_2 = np.clip(b_2 - lr * g_b2, -1, 1)
		b_3 = np.clip(b_3 - lr * g_b3, -1, 1)
		b_4 = np.clip(b_4 - lr * g_b4, -1, 1)
		b_5 = np.clip(b_5 - lr * g_b5, -1, 1)
		b_6 = np.clip(b_6 - lr * g_b6, -1, 1)

		# binarize
		W_1b = binarize(W_1)
		W_2b = binarize(W_2)
		W_3b = binarize(W_3)
		W_4b = binarize(W_4)
		W_5b = binarize(W_5)
		W_6b = binarize(W_6)

		b_1b = binarize(b_1)
		b_2b = binarize(b_2)
		b_3b = binarize(b_3)
		b_4b = binarize(b_4)
		b_5b = binarize(b_5)
		b_6b = binarize(b_6)

		# learning rate decay
		lr = lr * RATE_DECAY

print("Training finished")











