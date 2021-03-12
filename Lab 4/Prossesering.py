import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


"""KONSTANTER"""

fc_hp = 0.7
fc_lp = 4

filename = 'test_puls_rgb.txt'



"""DATAIMPORT"""
file = open(filename, 'r')
    data = file.read()

data_r = data[:,0]
data_g = data[:,1]
data_b = data[:,2]

plt.plot(data_r, 'r')
plt.plot(data_g, 'g')
plt.plot(data_b, 'b')
plt.title('original')
plt.show()



"""FILTRERING"""

soshp = signal.butter(10, fc_hp, 'hp', fs = fs, output='sos')
soslp = signal.butter(10, fc_lp, 'lp', fs = fs, output='sos')

data_r = signal.sosfilt(soshp, data_r)
data_r = signal.sosfilt(soslp, data_r)

data_g = signal.sosfilt(soshp, data_g)
data_g = signal.sosfilt(soslp, data_g)

data_b = signal.sosfilt(soshp, data_b)
data_b = signal.sosfilt(soslp, data_b)

