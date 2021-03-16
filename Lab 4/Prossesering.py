import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import csv


"""KONSTANTER"""

fc_hp = 0.5
fc_lp = 3.5
filnavn = 'tirsdag_test6_64.txt'
colors = ['r', 'g', 'b']
i = 0
upsample_factor = 16
steps_away = 5*upsample_factor
steps_window = 50*upsample_factor
fs = 40
lengde = 10


data_r = []
data_g = []
data_b = []


"""FUNKSJONER"""

def normalize(vec):
    maximum = np.amax(vec)
    return (vec/maximum)

def fourier(vector):
    sp = np.fft.fft(vector)
    freq = np.fft.fftfreq(vector.shape[-1])
    return abs(sp.real[0:int(len(sp)/2)]), freq[0:int(len(sp)/2)]

def SNR(sp, freq):
    index = np.argmax(sp)
    A_sig = sp[index]
    index += +steps_away
    A_noise = np.mean(sp[index:index+steps_window])
    SNR = A_sig/A_noise
    return SNR
    
def finn_puls(sp, freq):
    peak = np.argmax(sp)
    puls = abs(freq[peak]*fs*60)
    return puls


#Plotte SNR i rød
def plot_SNR(sp, freq):
    global i
    index=np.argmax(sp)+steps_away
    mean = np.mean(sp[index:index+steps_window])
    mean_vec = [mean]*steps_window
    plt.plot(freq[index:index+steps_window], mean_vec, colors[i])
    i += 1
    return

def upsample(array):
    array_new = signal.resample(array,upsample_factor*fs*lengde)
    return array_new


"""DATAIMPORT"""
with open(filnavn,'r') as myfile:
    data = csv.reader(myfile, delimiter= ' ')
    for line in data:
        data_r.append(float(line[0]))
        data_g.append(float(line[1]))
        data_b.append(float(line[2]))


data_r =  np.array(data_r)- np.mean(data_r)
data_g =  np.array(data_g) -np.mean(data_g)
data_b =  np.array(data_b) -np.mean(data_b)



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



data_r = upsample(normalize(data_r))
data_g = upsample(normalize(data_g))
data_b = upsample(normalize(data_b))

plt.plot(data_r, 'r')
plt.plot(data_g, 'g')
plt.plot(data_b, 'b')
plt.title('Upsamplet')
plt.show()


data_r = np.hstack((data_r, np.zeros(fs*lengde*upsample_factor*8)))
data_g = np.hstack((data_g, np.zeros(fs*lengde*upsample_factor*8)))                   
data_b = np.hstack((data_b, np.zeros(fs*lengde*upsample_factor*8)))

plt.plot(data_r, 'r')
plt.plot(data_g, 'g')
plt.plot(data_b, 'b')
plt.title('Zeropadding')
plt.show()

fs = 40*upsample_factor

"""PROSSESERING"""


#Fouriertransformasjon
sp_r, freq_r = fourier(data_r)
sp_g, freq_g = fourier(data_g)
sp_b, freq_b = fourier(data_b)

#Finnner pulsen basert på de ulike kanalene
puls_r = finn_puls(sp_r, freq_r)
puls_g = finn_puls(sp_g, freq_g)
puls_b = finn_puls(sp_b, freq_b)
print('puls:', puls_r, puls_g, puls_b)

#Finnner SNR i de ulike kanalene
SNR_r = SNR(sp_r,freq_r)
SNR_g = SNR(sp_g,freq_g)
SNR_b = SNR(sp_b,freq_b)
print('SNR:',SNR_r, SNR_g, SNR_b)

#plotter gjennomsnittet av støyen
plot_SNR(sp_r, freq_r)
plot_SNR(sp_g, freq_g)
plot_SNR(sp_b, freq_b)

#Plotte frekvensspekteret til rød
plt.plot(freq_r, abs(sp_r.real),'r')
plt.plot(freq_g, abs(sp_g.real),'g')
plt.plot(freq_b, abs(sp_b.real),'b')
plt.title('Fouriertransformasjon')
plt.show()
