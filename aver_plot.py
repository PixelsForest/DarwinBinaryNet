#!/usr/bin/python
# -*- coding:utf8 -*-
"""This file is used to average the data of 50 iterations"""
import os
import numpy as np
import matplotlib.pyplot as plt

file_index = 0


# load the data
print("data loading...")

while os.path.exists('data/main_toy/IB_IXT_BIN_ITERATION%s.npy' % file_index):
    exec("IB_IXT_BIN_ITERATION%s = np.load('data/main_toy/IB_IXT_BIN_ITERATION%s.npy')" % (file_index, file_index))
    exec("IB_IYT_BIN_ITERATION%s = np.load('data/main_toy/IB_IYT_BIN_ITERATION%s.npy')" % (file_index, file_index))
    file_index += 1

IB_epoch_color_BIN = np.load('data/main_toy/IB_epoch_color_BIN.npy')

print("successfully loaded.")


# averaging
print("doing average...")

IB_IXT_BIN = np.zeros(IB_IXT_BIN_ITERATION0.shape)
IB_IYT_BIN = np.zeros(IB_IYT_BIN_ITERATION0.shape)

for i in range(file_index):
    exec("IB_IXT_BIN += IB_IXT_BIN_ITERATION%s / file_index" % i)
    exec("IB_IYT_BIN += IB_IYT_BIN_ITERATION%s / file_index" % i)

print("averaging finished.")


# plot
font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 16}
cmap = plt.cm.get_cmap('gnuplot')

fig = plt.figure()
for i in range(len(IB_IXT_BIN)):
    plt.plot(IB_IXT_BIN[i], IB_IYT_BIN[i], marker='o', linestyle='-', markersize=12,
                 markeredgewidth=0.01, linewidth=0.2, color=cmap(IB_epoch_color_BIN[i]))

ax = plt.gca()
ax.set_xticks(np.linspace(1, 13, 13))
ax.set_yticks(np.linspace(0, 1, 6))
# size of xytick font
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

ax.set_xlabel('I(X;T)', fontdict=font)
ax.set_ylabel('I(Y;T)', fontdict=font)
plt.title('IB_BIN')
plt.savefig("data/main_toy/IB_BIN_AVERAGED.png")
plt.show()
