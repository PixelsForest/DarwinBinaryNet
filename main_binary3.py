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
y = y # label still 1 0, in order to fit softmax
print ('Loaded')

# parameters
LAYER_SIZE = [10, 7, 5, 4, 3]
EPOCHS = 8000
BATCH_SIZE = 512
lr = 0.0004
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


def batch_averager(x):
	# input: batchsize*D ; output: 1*D
	N = np.size(x, 0)
	return 1./N * np.sum(x, axis=0)


def batch_var(x, mu):
	# x: batchsize*D; mu: 1*D i.e.the average of x ; output: 1*D the variance
	N = np.size(x, 0)
	return 1./ N * np.sum((x - mu) * (x - mu), axis=0)


def x_hat(x, var, mu):
	# x_hat: batchsize*D
	deno = np.sqrt(var + 1e-9)
	return (x - mu)/deno


def x_BN(x, gamma, beta):
	# gamma: 1*D ; beta: 1*D
	return gamma * x + beta


def BP_BN(d_y, x, x_hat, mu, var, gamma, beta):
	m = np.size(x, 0)
	d_xhat = d_y * gamma # m*D
	d_var = np.sum(-0.5 * (d_xhat * (x - mu)) * np.power(var + 1e-3, -1.5), axis=0) # 1*D
	d_mu = np.sum(-d_xhat * np.power(var + 1e-3, -0.5), axis=0) + np.sum(-2./m * (x - mu) * d_var, axis=0) # 1*D
	d_x = d_xhat * np.power(var + 1e-3, -0.5) + 2./m * d_var * (x - mu) + 1./m * d_mu # m*D
	d_gamma = np.sum(d_y * x_hat, axis=0) # 1*D
	d_beta = np.sum(d_y, axis=0) # 1*D
	return d_x, d_gamma, d_beta

def softmax(z):
	# input: N * layer size ; output:N * layer size
	num = np.exp(z)
	denm = np.sum(np.exp(z), axis=1, keepdims=True)
	a = num / denm
	return a


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

beta_1 = np.random.randn(LAYER_SIZE[0])
beta_2 = np.random.randn(LAYER_SIZE[1])
beta_3 = np.random.randn(LAYER_SIZE[2])
beta_4 = np.random.randn(LAYER_SIZE[3])
beta_5 = np.random.randn(LAYER_SIZE[4])
beta_6 = np.random.randn(2)

gamma_1 = np.random.randn(LAYER_SIZE[0])
gamma_2 = np.random.randn(LAYER_SIZE[1])
gamma_3 = np.random.randn(LAYER_SIZE[2])
gamma_4 = np.random.randn(LAYER_SIZE[3])
gamma_5 = np.random.randn(LAYER_SIZE[4])
gamma_6 = np.random.randn(2)

C = 0.
# Adam optimizer
step = 0

m_W1 = [0.]
m_W2 = [0.]
m_W3 = [0.]
m_W4 = [0.]
m_W5 = [0.]
m_W6 = [0.]

v_W1 = [0.]
v_W2 = [0.]
v_W3 = [0.]
v_W4 = [0.]
v_W5 = [0.]
v_W6 = [0.]

m_beta1 = [0.]
m_beta2 = [0.]
m_beta3 = [0.]
m_beta4 = [0.]
m_beta5 = [0.]
m_beta6 = [0.]

v_beta1 = [0.]
v_beta2 = [0.]
v_beta3 = [0.]
v_beta4 = [0.]
v_beta5 = [0.]
v_beta6 = [0.]

m_gamma1 = [0.]
m_gamma2 = [0.]
m_gamma3 = [0.]
m_gamma4 = [0.]
m_gamma5 = [0.]
m_gamma6 = [0.]

v_gamma1 = [0.]
v_gamma2 = [0.]
v_gamma3 = [0.]
v_gamma4 = [0.]
v_gamma5 = [0.]
v_gamma6 = [0.]

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
for i in range(EPOCHS):
	# report training


	for j in range(int(np.size(y, 0)/BATCH_SIZE)):
		# training step
		step += 1

		#input layer a_0
		a_0 = F[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :]

		s_1 = np.dot(a_0, W_1b)
		mu_1 = batch_averager(s_1)
		var_1 = batch_var(s_1, mu_1)
		s_1hat = x_hat(s_1, var_1, mu_1)
		a_1 = x_BN(s_1hat, gamma_1, beta_1)
		a_1b = binarize(a_1)

		s_2 = np.dot(a_1b, W_2b)
		mu_2 = batch_averager(s_2)
		var_2 = batch_var(s_2, mu_2)
		s_2hat = x_hat(s_2, var_2, mu_2)
		a_2 = x_BN(s_2hat, gamma_2, beta_2)
		a_2b = binarize(a_2)

		s_3 = np.dot(a_2b, W_3b)
		mu_3 = batch_averager(s_3)
		var_3 = batch_var(s_3, mu_3)
		s_3hat = x_hat(s_3, var_3, mu_3)
		a_3 = x_BN(s_3hat, gamma_3, beta_3)
		a_3b = binarize(a_3)

		s_4 = np.dot(a_3b, W_4b)
		mu_4 = batch_averager(s_4)
		var_4 = batch_var(s_4, mu_4)
		s_4hat = x_hat(s_4, var_4, mu_4)
		a_4 = x_BN(s_4hat, gamma_4, beta_4)
		a_4b = binarize(a_4)

		s_5 = np.dot(a_4b, W_5b)
		mu_5 = batch_averager(s_5)
		var_5 = batch_var(s_5, mu_5)
		s_5hat = x_hat(s_5, var_5, mu_5)
		a_5 = x_BN(s_5hat, gamma_5, beta_5)
		a_5b = binarize(a_5)

		# output layer a_6
		s_6 = np.dot(a_5b, W_6b)
		mu_6 = batch_averager(s_6)
		var_6 = batch_var(s_6, mu_6)
		s_6hat = x_hat(s_6, var_6, mu_6)
		a_6 = x_BN(s_6hat, gamma_6, beta_6)
		# softmax layer
		A_6 = softmax(a_6)

		# gradients
		g_a6 = 1. / BATCH_SIZE *(A_6 - y[j*BATCH_SIZE:(j+1)*BATCH_SIZE, :]) # (BATCH_SIZE, layer_size)
		g_s6, g_gamma6, g_beta6 = BP_BN(g_a6, s_6, s_6hat, mu_6, var_6, gamma_6, beta_6)

		g_a5b = np.dot(g_s6, W_6b.T)
		g_a5 = g_a5b * STD(a_5)
		g_s5, g_gamma5, g_beta5 = BP_BN(g_a5, s_5, s_5hat, mu_5, var_5, gamma_5, beta_5)

		g_a4b = np.dot(g_s5, W_5b.T)
		g_a4 = g_a4b * STD(a_4)
		g_s4, g_gamma4, g_beta4 = BP_BN(g_a4, s_4, s_4hat, mu_4, var_4, gamma_4, beta_4)

		g_a3b = np.dot(g_s4, W_4b.T)
		g_a3 = g_a3b * STD(a_3)
		g_s3, g_gamma3, g_beta3 = BP_BN(g_a3, s_3, s_3hat, mu_3, var_3, gamma_3, beta_3)

		g_a2b = np.dot(g_s3, W_3b.T)
		g_a2 = g_a2b * STD(a_2)
		g_s2, g_gamma2, g_beta2 = BP_BN(g_a2, s_2, s_2hat, mu_2, var_2, gamma_2, beta_2)

		g_a1b = np.dot(g_s2, W_2b.T)
		g_a1 = g_a1b * STD(a_1)
		g_s1, g_gamma1, g_beta1 = BP_BN(g_a1, s_1, s_1hat, mu_1, var_1, gamma_1, beta_1)

		g_W1 = np.dot(a_0.T, g_s1)
		g_W2 = np.dot(a_1b.T, g_s2)
		g_W3 = np.dot(a_2b.T, g_s3)
		g_W4 = np.dot(a_3b.T, g_s4)
		g_W5 = np.dot(a_4b.T, g_s5)
		g_W6 = np.dot(a_5b.T, g_s6)

		#update
		W_1 = np.clip(W_1 - lr * Adam(m_W1, v_W1, g_W1, step), -1, 1)
		W_2 = np.clip(W_2 - lr * Adam(m_W2, v_W2, g_W2, step), -1, 1)
		W_3 = np.clip(W_3 - lr * Adam(m_W3, v_W3, g_W3, step), -1, 1)
		W_4 = np.clip(W_4 - lr * Adam(m_W4, v_W4, g_W4, step), -1, 1)
		W_5 = np.clip(W_5 - lr * Adam(m_W5, v_W5, g_W5, step), -1, 1)
		W_6 = np.clip(W_6 - lr * Adam(m_W6, v_W6, g_W6, step), -1, 1)

		gamma_1 = gamma_1 - lr * Adam(m_gamma1, v_gamma1, g_gamma1, step)
		gamma_2 = gamma_2 - lr * Adam(m_gamma2, v_gamma2, g_gamma2, step)
		gamma_3 = gamma_3 - lr * Adam(m_gamma3, v_gamma3, g_gamma3, step)
		gamma_4 = gamma_4 - lr * Adam(m_gamma4, v_gamma4, g_gamma4, step)
		gamma_5 = gamma_5 - lr * Adam(m_gamma5, v_gamma5, g_gamma5, step)
		gamma_6 = gamma_6 - lr * Adam(m_gamma6, v_gamma6, g_gamma6, step)

		beta_1 = beta_1 - lr * Adam(m_beta1, v_beta1, g_beta1, step)
		beta_2 = beta_2 - lr * Adam(m_beta2, v_beta2, g_beta2, step)
		beta_3 = beta_3 - lr * Adam(m_beta3, v_beta3, g_beta3, step)
		beta_4 = beta_4 - lr * Adam(m_beta4, v_beta4, g_beta4, step)
		beta_5 = beta_5 - lr * Adam(m_beta5, v_beta5, g_beta5, step)
		beta_6 = beta_6 - lr * Adam(m_beta6, v_beta6, g_beta6, step)

		# binarize
		W_1b = binarize(W_1)
		W_2b = binarize(W_2)
		W_3b = binarize(W_3)
		W_4b = binarize(W_4)
		W_5b = binarize(W_5)
		W_6b = binarize(W_6)

		# learning rate decay
		lr = lr * RATE_DECAY

		#if i % 100 == 0 and i > 0:
			#print("EPOCH: %s, learning rate: %s" %(i, lr))
			#print("a_6", a_6[100:105])
			#print("s_6", s_6[100:105])
			#print("a_5b", a_5b[100:105])
			#print("W_6:", W_6)
			#print("g_W6", g_W6)
			#print("a_6", a_6[100:110])
			#print("gamma_6:", gamma_6)
			#print("g_gamma6", g_gamma6)
			#print("beta_6", beta_6)
			#print("g_beta6", g_beta6)

	# testing it
	if (i-1) % 100 == 0: # i % 100 == 0 and i > 0
		print("epoch : %s " % i)
		a_0 = F

		s_1 = np.dot(a_0, W_1b)
		mu_1 = batch_averager(s_1)
		var_1 = batch_var(s_1, mu_1)
		s_1hat = x_hat(s_1, var_1, mu_1)
		a_1 = x_BN(s_1hat, gamma_1, beta_1)
		a_1b = binarize(a_1)

		s_2 = np.dot(a_1b, W_2b)
		mu_2 = batch_averager(s_2)
		var_2 = batch_var(s_2, mu_2)
		s_2hat = x_hat(s_2, var_2, mu_2)
		a_2 = x_BN(s_2hat, gamma_2, beta_2)
		a_2b = binarize(a_2)

		s_3 = np.dot(a_2b, W_3b)
		mu_3 = batch_averager(s_3)
		var_3 = batch_var(s_3, mu_3)
		s_3hat = x_hat(s_3, var_3, mu_3)
		a_3 = x_BN(s_3hat, gamma_3, beta_3)
		a_3b = binarize(a_3)

		s_4 = np.dot(a_3b, W_4b)
		mu_4 = batch_averager(s_4)
		var_4 = batch_var(s_4, mu_4)
		s_4hat = x_hat(s_4, var_4, mu_4)
		a_4 = x_BN(s_4hat, gamma_4, beta_4)
		a_4b = binarize(a_4)

		s_5 = np.dot(a_4b, W_5b)
		mu_5 = batch_averager(s_5)
		var_5 = batch_var(s_5, mu_5)
		s_5hat = x_hat(s_5, var_5, mu_5)
		a_5 = x_BN(s_5hat, gamma_5, beta_5)
		a_5b = binarize(a_5)

		s_6 = np.dot(a_5b, W_6b)
		mu_6 = batch_averager(s_6)
		var_6 = batch_var(s_6, mu_6)
		s_6hat = x_hat(s_6, var_6, mu_6)
		a_6 = x_BN(s_6hat, gamma_6, beta_6)
		# softmax layer
		A_6 = softmax(a_6)

		# loss
		L = 1. / y.shape[0] * np.sum(-np.log(A_6 + 1e-8) * y)
		print("the loss：%s" % L)
		#print(a_6[900:1000])

		predict = (np.amax(A_6, axis=1, keepdims=True) == A_6)
		predict = np.where(predict, 1., 0.)
		accur = float(np.sum(predict * y)) / y.shape[0]
		print('the accuracy of DNN is: %s' % accur)

print("Training finished")