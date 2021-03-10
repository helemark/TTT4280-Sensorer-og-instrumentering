import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


'''Konstanter'''

f0 = 2.413e10    #denne tregner vi vel forsåvidt ikke
fs = 31250          #samplingsfrekvensen på Pi'en
c = 3e8
k = 100


"""Utdelt kode"""
def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
    return sample_period, data


# Import data from bin file
sample_period, data = raspi_import('fredag_bil3_2ms.bin')

#data = signal.detrend(data, axis=0)  # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix



"""HELENES KODE"""

def fourier(vector):
    sp = np.fft.fft(vector)
    freq = np.fft.fftfreq(vector.shape[-1])
    return sp, freq

def SNR(index, vec):
    


def speed(sp, freq):                                            #Regner ut farten til objektet ut fra frekvenskomponenten med størst amplitude
    index = np.argmax(abs(sp))
    fd = abs(freq[index]*fs)
    print('fd:', fd)
    speed = (fd*c)/(2*f0)
    return speed

def butter_bandstop_filter(data, lowcut, highcut, fs, order):   #Hentet fra stack overflow
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    i, u = signal.butter(order, [low, high], btype='bandstop')
    y = signal.lfilter(i, u, data)
    return y
    


'''HER BEGYNNER SELVE KODEN'''

data_0 = data[:,0][k*3:]-np.mean(data[:,0])                     #tar ikke med de første samplene, for de har ekstra mye støy
data_1 = data[:,1][k*3:]-np.mean(data[:,1])                     #tar ikke med de første samplene, for de har ekstra mye støy


plt.plot(data_1)
plt.plot(data_0)
plt.title('original')
plt.show()


"""FILTRERING"""

soshp = signal.butter(10, 80, 'hp', fs = fs, output='sos')      #høypassfilter med fc=80Hz
soslp = signal.butter(10, 800, 'lp', fs = fs, output='sos')     #lavpassfilter med fc=800Hz

data_0 = signal.sosfilt(soshp, data_0)
data_0 = signal.sosfilt(soslp, data_0)
data_0 = butter_bandstop_filter(data_0, 650, 690, fs, 4)        # vi hadde en veldig tydelig støykomponent i dette området som vi måtte fjerne

data_1 = signal.sosfilt(soshp, data_1)
data_1 = signal.sosfilt(soslp, data_1)
data_1 = butter_bandstop_filter(data_1, 650, 690, fs, 4)        # vi hadde en veldig tydelig støykomponent i dette området som vi måtte fjerne



plt.plot(data_1)
plt.plot(data_0)
plt.title('Etter filtrering')
plt.show()                                                      #plotter resultet etter filtreringen for å observere at alt ser bra ut


"""SIGNALBEHANDLING"""
sp, freq = fourier(1j*data_0 + data_1)                          #kompleks fouriertransformasjon, ikke symmetrisk!
plt.plot(freq, abs(sp.real),'r') 
plt.title('IF_I+jIF_Q')
#plt.yscale('log')     
plt.show()


speed = speed(sp, freq)
print(speed)


