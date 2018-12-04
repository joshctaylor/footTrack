# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:15:01 2017
@author: jct1c13
"""
# add the tracking and audio extraction into same script - charge £££
# use lsits to allow a 
# input grid
# export using opencv
import wave
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 

wav1 = wave.open('ov_A8S12.wav')
wav2 = wave.open('aft_A8S12.wav')
wav3 = wave.open('fore_A8S12.wav')

signal1 = wav1.readframes(-1)
signal1 = np.fromstring(signal1, 'Int16')
signal1 = signal1 / np.mean(signal1)  # normalise

signal2 = wav2.readframes(-1)
signal2 = np.fromstring(signal2, 'Int16')
signal2 = signal2 / np.mean(signal2)  # normalise

signal3 = wav3.readframes(-1)
signal3 = np.fromstring(signal3, 'Int16')
signal3 = signal3 / np.mean(signal3)  # normalise

fs = wav1.getframerate()
 

plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(signal1)
plt.ylabel('aft')

plt.subplot(3, 1, 2)
plt.plot(signal2)
plt.ylabel('fore')

plt.subplot(3, 1, 3)
plt.plot(signal3)
plt.ylabel('overview')
       
plt.show()


# corr = np.correlate(signal1, signal2)
corr  = signal.correlate(signal1, signal2, mode='valid', method = 'fft')
delay1 = corr.argmax() / fs
corr  = signal.correlate(signal1, signal3, mode='valid', method = 'fft')
delay2 = corr.argmax() / fs