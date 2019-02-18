import numpy as np
import copy
import scipy.io as sio
import information_toolbox as it
import matplotlib.pyplot as plt
"""One layer +-1 network. Weight evolve randomly"""

# read the data
print ('Loading Data...')
data = sio.loadmat('var_u.mat')
F = data['F']  # 4096*12
y = data['y'].T  # 4096*1
# -1 1
F = 2*np.array(F) - 1.
y = 2*np.array(y) - 1.
print ('Loaded')


def binarize(x):
	bi = np.clip(x, -1., 0.999)
	bi = bi + 1.
	bi = (bi.astype(np.int))*2 - 1
	return bi.astype(np.float)

def random_adjust(w, sigma):
	# randomly choose one value in matrix w, then modify it
	x = copy.deepcopy(w)
	l = np.random.randint(0, x.shape[0])
	c = np.random.randint(0, x.shape[1])
	x[l][c] = x[l][c] + np.random.normal(0., sigma)
	return x

def plot_and_save(axis_1, axis_2, iter, NTH, STH):
	# plot and save the figure
	font = {'family': 'serif',
			'color': 'black',
			'weight': 'normal',
			'size': 16}
	cmap = plt.cm.get_cmap('viridis')

	# Information plane
	fig = plt.figure()
	for i in range(len(axis_1)):
		plt.plot(axis_1[i], axis_2[i], marker='o', linestyle='-', markersize=10,
				 markeredgewidth=0.01, linewidth=0.2, color=cmap(float(NTH[i]-2048.) / (0.5*F.shape[0])))

	ax = plt.gca()
	ax.set_xticks(np.linspace(1, 13, 13))
	ax.set_yticks(np.linspace(0, 1, 6))
	# size of xytick font
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)

	ax.set_xlabel('I(X;T)', fontdict=font)
	ax.set_ylabel('I(Y;T)', fontdict=font)
	plt.title('main_random %s' % NUM_CRT)
	plt.savefig("data/main_random/36/IB_BIN_ITERATION%s.png" % iter)
	plt.close()

	# Training process
	fig = plt.figure()
	plt.plot(np.log(STH), NTH, marker='x', linestyle='-', markersize=5, linewidth=0.2, color='r')
	ax = plt.gca()
	ax.set_xlabel('ln(steps)', fontdict=font)
	ax.set_ylabel('correct number', fontdict=font)
	plt.title('main_random ')
	plt.savefig("data/main_random/36/TRAIN_ITERATION%s.png" % iter)
	plt.close()



LAYERSIZE = 36
PROB = 0.7 # the prob to decide which w to update
SIGMA = 0.01
INFORMATION = True
PLOT = True
ITERATION = 50


for iteration in range(ITERATION):
	# initialize
	W_1 = np.zeros((F.shape[1] + 1, LAYERSIZE))
	W_2 = np.zeros((LAYERSIZE + 1, y.shape[1]))
	w_1 = np.zeros((F.shape[1] + 1, LAYERSIZE))
	w_2 = np.zeros((LAYERSIZE + 1, y.shape[1]))

	NUM_CRT = 0.
	STEP = 0
	DATA = {
		"axis_1": [],  # [[IXT], [IXT], ...]
		"axis_2": [],  # [[IYT], [IYT], ...]
		"NUM_CRT_HISTORY":[],   #   [2118, 2118, 2490, ...]
		"STEPS_HISTORY":[], #  [1, 2, 5, ...]
	}
	W_HISTORY = []  # [[W_1, W_2], [W_1, W_2], ...] This can be pretty big so I separate it with DATA


	while True:
		if NUM_CRT >= 4000:
			break

		if np.random.random() < PROB:
			w_1 = random_adjust(W_1, SIGMA)
		else:
			w_2 = random_adjust(W_2, SIGMA)

		a_0 = F
		b_1 = np.ones(F.shape[0])
		z_1 = np.dot(np.column_stack([a_0, b_1]), w_1)
		a_1 = binarize(z_1)
		b_2 = np.ones(F.shape[0])
		z_2 = np.dot(np.column_stack([a_1, b_2]), w_2)
		a_2 = binarize(z_2)

		num_crt = np.sum(a_2 == y)

		STEP += 1
		if num_crt >= NUM_CRT:
			W_1 = copy.deepcopy(w_1)
			W_2 = copy.deepcopy(w_2)
			# check if num_crt is bigger than NUM_CRT
			check = num_crt - NUM_CRT
			NUM_CRT = num_crt
			print("The correct prediction number now: %s" % num_crt)
			if INFORMATION:
				I_XT1_BIN = it.bin_calc_information(F, a_1, -1., 1., 2)
				I_YT1_BIN = it.bin_calc_information(y, a_1, -1., 1., 2)

				DATA["axis_1"].append([I_XT1_BIN])
				DATA["axis_2"].append([I_YT1_BIN])
				DATA["NUM_CRT_HISTORY"].append(NUM_CRT)
				DATA["STEPS_HISTORY"].append(STEP)
				W_HISTORY.append([W_1, W_2])

				if PLOT:
					plot_and_save(DATA["axis_1"], DATA["axis_2"],
								  iteration, DATA["NUM_CRT_HISTORY"], DATA["STEPS_HISTORY"])
				if NUM_CRT >= 3500 and check:
					np.save('data/main_random/36/DATA_ITERATION%s.npy' % iteration, DATA)
					np.save('data/main_random/36/W_ITERATION%s.npy' % iteration, W_HISTORY)