import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import csv


"""KONSTANTER"""

fc_hp = 0.5
fc_lp = 6
fs = 40
filnavn = 'test_puls_output.csv'



data_r = []
data_g = []
data_b = []


"""FUNKSJONER"""

def fourier(vector):
    sp = np.fft.fft(vector)
    freq = np.fft.fftfreq(vector.shape[-1])
    return sp, freq




"""DATAIMPORT"""
with open(filnavn,'r') as myfile:
    data = csv.reader(myfile, delimiter= ' ')
    for line in data:
        data_r.append(float(line[0]))
        data_g.append(float(line[1]))
        data_b.append(float(line[2]))

print(len(data_r))
plt.plot(data_r, 'r')
plt.plot(data_g, 'g')
plt.plot(data_b, 'b')
plt.title('Original')
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

plt.plot(data_r, 'r')
plt.plot(data_g, 'g')
plt.plot(data_b, 'b')
plt.title('Filtrert')
plt.show()


"""PROSSESERING"""

sp_r, freq_r = fourier(data_r)
plt.plot(freq_r, abs(sp_r.real),'r')
plt.title('Fouriertransformasjon')
plt.show()

peak = np.argmax(sp_r)
print(peak)
puls = abs(freq_r[peak]*fs*60)

print(puls)

