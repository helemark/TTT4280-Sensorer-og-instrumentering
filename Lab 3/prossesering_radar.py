import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal





'''Konstanter'''

f0 = 2.413e10    #denne tregner vi vel forsåvidt ikke
fs = 31250          #samplingsfrekvensen på Pi'en
c = 3e8
k=100

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


'''Helene's Funksjoner'''


def fourier(vector):
    sp = np.fft.fft(vector)
    freq = np.fft.fftfreq(vector.shape[-1])
    return sp, freq


def speed(sp, freq):
    index = np.argmax(abs(sp))
    fd = abs(freq[index]*fs)
    print('fd:', fd)
    speed = (fd*c)/(2*f0)
    return speed

def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = signal.butter(order, [low, high], btype='bandstop')
    y = signal.lfilter(i, u, data)
    return y
    


'''
def speed_of_time(vector, k, fs):
    n = int(len(vector)/k)
    window = np.hanning(n*2)
    speed = []
    for i in range(1, k):
        sp, freq = fourier(vector[int(i*n-n):int(i*n+n)]*window)
        fd = fs*abs(freq[np.argmax(abs(sp.real))])
        print(fd)
        speed.append(fd*c/(2*f0))
    return speed  
'''



'''HER BEGYNNER SELVE KODEN'''

data_0 = data[:,0][k*3:]-np.mean(data[:,0])
data_1 = data[:,1][k*3:]-np.mean(data[:,1])


plt.plot(data_1)
plt.plot(data_0)
plt.title('original')
plt.show()


"""FILTRERING"""

soshp = signal.butter(10, 80, 'hp', fs = fs, output='sos')
soslp = signal.butter(10, 800, 'lp', fs = fs, output='sos')

data_0 = signal.sosfilt(soshp, data_0)
data_0 = signal.sosfilt(soslp, data_0)
data_0 = butter_bandstop_filter(data_0, 650, 690, fs, 4)

data_1 = signal.sosfilt(soshp, data_1)
data_1 = signal.sosfilt(soslp, data_1)
data_1 = butter_bandstop_filter(data_1, 650, 690, fs, 4)



plt.plot(data_1)
plt.plot(data_0)
plt.title('Etter filtrering')
plt.show()

sp0, freq0 = fourier(data_0)
sp1, freq1 = fourier(data_1)

'''
plt.plot(freq0, sp0.real, 'b')
plt.plot(freq1, sp1.real, 'r')
plt.title('Spectre')
plt.show()
'''

sp, freq = fourier(1j*data_0 + data_1)
plt.plot(freq, abs(sp.real),'r')
plt.title('IF_I+jIF_Q')
sp, freq = fourier(data_1)
plt.plot(freq, abs(sp.real),'b')
plt.show()


'''
plt.plot(sp)
plt.show()
'''

speed = speed(sp, freq)
print(speed)

