import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import tkinter as tk


#funksjon som henter data fra .bin-fila
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


#Importerer data fra fila
sample_period, data = raspi_import('DATA_OUTPUT.bin')
#print(sample_period)

sample_period *= 1e-6  # change unit to micro seconds

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix

'''LITT DIVERSE KONSTANTER'''
Fs = 31250 #samplingsfrekvensen
N  = 20000 #lengde brukt på første parameter i korrelasjonen
N_short = 2000 #lengden vi bruker på andre parameter
d = 0.064 #avstand mellom mic'ene i meter
upsample_factor = 16 #faktor vi upsampler med
c = 343 #lydhastigheten isj

print('Max delay in samples:', int(Fs*upsample_factor*d/c)) #pri


#korrelasjon funksjon fordi det ser penere ut
def correlation(vec_1, vec_2):
    corr = np.correlate(vec_1, vec_2, "valid")
    return corr

#Brukes ikke, bruker np.argmax(vec) istendenfor
def corr_max(corr):
    maximum = float(np.max(corr))
    i=0
    while(i<len(corr)):
        if abs(corr[i]) == maximum:
            return i
        else:
            i += 1

#Normalsiering, fordi det er best alltid <333
def normalize(vec):
    maximum = np.amax(vec)
    return (vec/maximum)


#Upsampling
def upsample(array):
    array_new = signal.resample(array,upsample_factor*Fs)
    return array_new

#Fjerner de 100 første og siste samplene, finner index'en til maxverdien i vektoren, og lager en ny vektor med lengde N, sentrert rundt maxverdien
def make_nice(array, N):
    for i in range(100):
        array = np.delete(array, 0)
        array = np.delete(array, len(array)-1)
    max_index = np.argmax(array)
    array_new = array[int(max_index-N/2):int(max_index+N/2)]
    return [array_new, max_index]

#Lager en ny vektor med lengde N, sentrert rundt mean_index
def make_nice_around(array, N, mean_index):
    return array[int(mean_index-N/2):int(mean_index+N/2)]


#Upsampler og fjerner DC-komponent
mean = np.mean(data[:,1:4])
data_1 = upsample(data[:,1])-mean
data_2 = upsample(data[:,2])-mean
data_3 = upsample(data[:,3])-mean



#Plotter dataen slik den er
plt.plot(data_1, 'blue')
plt.plot(data_2, 'green')
plt.plot(data_3, 'red')
plt.title('Original signals')
plt.show()

#Kutter signalene
data_1,i1 = make_nice(data_1, N)
data_2 = make_nice_around(data_2, N, i1+100) #legger til 100 fordi indexen er etter klipp
data_3 = make_nice_around(data_3, N, i1+100)

#plotter kuttede signaler
plt.plot(data_1, 'blue')
plt.plot(data_2, 'green')
plt.plot(data_3, 'red')
plt.title('Cut signals')
plt.show()





#FILTRERING SKJER HER   
soshp = signal.butter(10, 50, 'hp', fs = Fs, output='sos') #Høypassfilter
soslp = signal.butter(10, 1000, 'lp', fs = Fs, output='sos') #Lavpassfilter

data_1 = signal.sosfilt(soshp, data_1)
data_1 = signal.sosfilt(soslp, data_1)

data_2 = signal.sosfilt(soshp, data_2)
data_2 = signal.sosfilt(soslp, data_2)

data_3 = signal.sosfilt(soshp, data_3)
data_3 = signal.sosfilt(soslp, data_3)

#Plotter filtrert filtrert data
plt.plot(data_1, 'blue')
plt.plot(data_2, 'green')
plt.plot(data_3, 'red')
plt.title('Cut and filtered signals')
plt.show()

#De mindre arrayene som skal brukes i korrelasjonene!
data_1_short = make_nice_around(data_1, N_short, np.argmax(data_1))
data_2_short = make_nice_around(data_2, N_short, np.argmax(data_2))
data_3_short = make_nice_around(data_3, N_short, np.argmax(data_3))



'''CORRELATION TIME, COME'ON!'''

#KORRELASJON og autokorrelasjon skjer her!
corr21 = normalize(correlation(data_2, data_1_short))
corr31 = normalize(correlation(data_3, data_1_short))
corr32 = normalize(correlation(data_3, data_2_short))
autocorr11 = normalize(correlation(data_1, data_1_short))
autocorr22 = normalize(correlation(data_2, data_2_short))
autocorr33 = normalize(correlation(data_3, data_3_short))


#Finner differansen mellom makspuntindexene til korrelasjonene og autokorrelasjonene
# ---> FORSINKELSEN I SAMPLER
delay21 = np.argmax(autocorr11)-np.argmax(corr21)
delay31 = np.argmax(autocorr11)-np.argmax(corr31)
delay32 = np.argmax(autocorr22)-np.argmax(corr32)


'''
print('Argmax of corr ')
print(np.argmax(data_1))
print(np.argmax(data_2))
print(np.argmax(data_3))
'''

print()
print('Delay')
print('n21:',delay21)
print('n31:',delay31)
print('n32:',delay32)

teta = np.arctan(np.sqrt(3)*(delay21+delay32)/(delay21-delay31-2*delay32))
if(-delay21+delay31+2*delay32)<0:
    teta += np.pi

print()
print('Teta:', teta)

'''
#Plotting av korrelasjoner
plt.plot(corr21)
plt.title('corr21')
plt.show()

plt.plot(corr31)
plt.title('corr31')
plt.show()

plt.plot(corr32)
plt.title('corr32')
plt.show()

plt.plot(autocorr22)
plt.plot(autocorr11)
plt.title('autocorr')
plt.show()
'''






"""Jihaa, nå bli're kult"""


#Grafikkdel!!!! <3<3<3<3
def _create_circle_arc(self, x, y, r, **kwargs):
    if "start" in kwargs and "end" in kwargs:
        kwargs["extent"] = kwargs["end"] - kwargs["start"]
        del kwargs["end"]
    return self.create_arc(x-r, y-r, x+r, y+r, **kwargs)
tk.Canvas.create_circle_arc = _create_circle_arc


#mine funksjoner
def mic(canvas,x,y, r):
   id = canvas.create_oval(x-r,y-r,x+r,y+r, fill = "black")
   return id

def circle(canvas,x, y, r):
   id = canvas.create_oval(x-r,y-r,x+r,y+r)
   return id

def angle(canvas, x, y, teta, r):
   x_end = x + np.cos(teta)*r
   y_end = y - np.sin(teta)*r
   id = canvas.create_line(x, y, x_end, y_end)
   return id

def show_angle(canvas, x, y, teta, radius):
   id = canvas.create_circle_arc(x, y, radius, outline = "red", start=0, end=int(180*teta/np.pi))
   return id

#Lager grunnstrukturen:
master = tk.Tk()
master.title("Microphones")
canvas_width = 400
canvas_height = 400

w = tk.Canvas(
    master, 
    width=canvas_width,
    height=canvas_height
    )


w.pack()

a = 150
r = 5
r_angle = 10

center_x = canvas_width/2
center_y = canvas_height/2
mic_1 = [canvas_width/2, canvas_height/2 - a]
mic_2 = [canvas_width/2 -np.sqrt(3)*a/2, canvas_height/2 + a/2]
mic_3 = [canvas_width/2 +np.sqrt(3)*a/2, canvas_height/2 + a/2]


mic(w, mic_1[0], mic_1[1], r)
mic(w, mic_2[0], mic_2[1], r)
mic(w, mic_3[0], mic_3[1], r)

circle(w, canvas_width/2, canvas_height/2, a)
mic(w, canvas_width/2, canvas_height/2, 1)

#linje til å markere teta = 0
w.create_line(center_x, center_y, center_x + a, center_y, width = 3)

#plotter teta 
angle(w, center_x, center_y, teta, a)
show_angle(w, center_x, center_y, teta, r_angle)
r_angle += 5

w.mainloop()


