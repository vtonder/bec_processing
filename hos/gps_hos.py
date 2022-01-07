import numpy as np
from matplotlib import pyplot as plt
from hos import bispectrum
from matplotlib import cm
from matplotlib.ticker import LinearLocator

class SIM_GPS:
    def __init__(self, files):
        self.gps_data = {}
        for file in files:
            self.gps_data.update({file:[]})

    def populate(self, direc):
        for file_name, data  in self.gps_data.items():
            with open(direc+file_name) as f:
                for line in f.readlines():
                    data.append(float(line))

#gps_file_names = ['I_CA_D_4bit.csv', 'I_CA_D_1bit.csv','I_PY_4bit.csv', 'I_PY_1bit.csv', 'I_PY_D_4bit.csv', 'I_PY_D_1bit.csv',
#                  'Q_CA_D_4bit.csv', 'Q_CA_D_1bit.csv','Q_CA_4bit.csv', 'Q_CA_1bit.csv', 'Q_L2CM_Dc_4bit.csv', 'Q_L2CM_Dc_1bit.csv']
#gps_file_names = ['I_CA_D_2bit.csv','I_PY_2bit.csv', 'I_PY_D_2bit.csv',
#                  'Q_CA_D_2bit.csv','Q_CA_2bit.csv', 'Q_L2CM_2bit.csv']
gps_file_names = ['I_CA_D_8bit_1412.csv','I_PY_8bit_1412.csv','I_PY_D_8bit_1412.csv']

gps_file_names = ['I_CA_D_1bit.csv']
gps = SIM_GPS(gps_file_names)
gps.populate('/home/vereese/git/phd_data/')

M = 1024
fs_p = 21.518 # MHz sampling rate for P and P(Y) code
fs_ca = 2.1518 # MHz sampling rate for C/A code
freq_res_p = fs_p/M
freq_res_ca = fs_ca/M
freq_p = np.arange(0,fs_p,freq_res_p)
freq_ca = np.arange(0,fs_ca,freq_res_ca)

bispectra = {}
for fn in gps_file_names:
    data = gps.gps_data[fn]
    data_len = len(data)
    print(fn, data_len)
    fft_len = int(M)
    records = int(np.floor(data_len / fft_len))
    M_2 = int(fft_len / 2)
    cum = np.zeros([records, M_2, M_2], dtype='complex_')
    data = np.asarray(data[0:int(fft_len * records)]).reshape(records, fft_len)

    b = bispectrum(data, method='direct')
    b.calc_bispectrum()
    np.save(fn+'_bispectrum',b.full_bispec)
    #b.calc_power_spectrum()
    #b.plot_bispectrum(name=fn)
    #bispectra.update({fn:b})
    #b.plot_power_spectrum(fn)
    #print(np.shape(b.full_bispec))
    #X = np.arange(1024)
    #Y = np.arange(1024)
    #X, Y = np.meshgrid(X, Y)
    #Z = np.abs(bispectra[fn].full_bispec)
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm,linewidth=0, antialiased=False)
    #plt.show()

# tested if 2 codes are the same
# tested if gps simulator consistently gives out same pattern for same config and it does
#a = np.abs(bispectra['Q_L2CM_Dc_8bit_1412.csv'].bispectrum_I)
#b = np.abs(bispectra['Q_CA_8bit_1412.csv'].bispectrum_I)
#c = list(np.sum(a - b, axis=0))
#if not any(c): print("They're identical")

# tested to see if PY data == PYxorD data
# result , the data is different but the power spectrum and the bispectrum is the same when 1 /2 / 4 bits are used
# when 8 bits are used, they are not the same.
#plt.figure(0)
#plt.plot(bispectra['./data/I_PY_1bit.csv'].power_spectrum,label='PY')
#plt.plot(bispectra['./data/I_PY_D_1bit.csv'].power_spectrum,label='PY xor D')
#plt.legend()
#plt.grid()
#plt.show()


'''for i in np.arange(0,2,2):
    fn_2bit = gps_file_names[i]
    fn_1bit = gps_file_names[i+1]
    plt.figure(i)
    plt.imshow(np.abs(bispectra[fn_2bit].full_bispec), aspect='auto', origin='lower')
    plt.title(fn_2bit)
    plt.figure(i+1)
    plt.imshow(np.abs(bispectra[fn_1bit].full_bispec), aspect='auto', origin='lower')
    plt.title(fn_1bit)
plt.show()'''

#bispectra['I_CA_D.csv'].plot_bispectrum(freq_p, 'i_py')
#np.save('gps_bispectrum', b.bispectrum)

"""FN = np.fft.fft(gps)

for k1 in np.arange(M_2):
    for k2 in np.arange(M_2):
        cum[:,k1,k2] = 1.0/M_2 * FN[:,k1]*FN[:,k2]*np.conj(FN[:,k1+k2])
cum2 = 1/10 * cum.sum(axis=0)

plt.figure(0)
#plt.plot(freq[0:500],cum2[:,])
plt.imshow(np.abs(FN), aspect='auto', origin='lower') #, aspect='auto', extent=[0,500,0,500])
#plt.xticks(freq[0:500])

plt.figure(1)
#plt.plot(freq[0:500],cum2[:,])
plt.imshow(np.abs(cum2), aspect='auto', origin='lower') #, aspect='auto', extent=[0,500,0,500])
#plt.xticks(freq[0:500])
plt.show()"""