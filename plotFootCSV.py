# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:18:54 2017

@author: jct1c13
"""

from matplotlib import pyplot as plt
import numpy as np

fileName = 'E:/aftereffets/a8d_04.mp4blur_mod.csv'
data = np.genfromtxt(fileName, delimiter=',', names=['frame', 'x', 'y', 'r'])

plt.ion()
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ln1 = plt.plot(data['frame'], data['x'], color='r', label='foot position')
ax2 = ax1.twinx()
ln2 = plt.plot(data['frame'], data['r'], color='b', label='blob radius')

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
plt.draw()

plt.ion()
plt.plot([1.6, 2.7])
plt.title("interactive test")
plt.xlabel("index")
ax = plt.gca()
ax.plot([3.1, 2.2])
plt.draw()


