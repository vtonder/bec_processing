import numpy as np
from matplotlib import pyplot as plt
from hos import bispectrum
import sys
from matplotlib import cm
from matplotlib.ticker import LinearLocator

DIRECTORY = '/home/vereese/git/phd_data/'
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

if len(sys.argv) > 1:
    gps_file_names = []
    for i in np.arange(1, len(sys.argv)):
        gps_file_names.append(str(sys.argv[i]))
else:
    gps_file_names = ['I_PY_1bit.csv', 'I_PY_D_1bit.csv', 'I_CA_D_1bit.csv','Q_CA_D_1bit.csv', 'Q_CA_1bit.csv','Q_L2CM_Dc_1bit.csv']

gps = SIM_GPS(gps_file_names)
gps.populate(DIRECTORY)

M = 1024
fs_p = 21.518 # MHz sampling rate for P and P(Y) code
fs_ca = 2.1518 # MHz sampling rate for C/A code
freq_res_p = fs_p/M
freq_res_ca = fs_ca/M
freq_p = np.arange(0,fs_p,freq_res_p)
freq_ca = np.arange(0,fs_ca,freq_res_ca)

bispectra = {}
for i, fn in enumerate(gps_file_names):
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
    #np.save(DIRECTORY+fn[:-4]+'_bispec',b.full_bispec)
    b.calc_power_spectrum()
    b.plot_bispectrum(name=fn)
    #bispectra.update({fn:b})
    b.plot_power_spectrum(i,fn)
    #print(np.shape(b.full_bispec))
    plt.show()

# tested if 2 codes are the same
# tested if gps simulator consistently gives out same pattern for same config and it does
#a = np.abs(bispectra['Q_L2CM_Dc_8bit_1412.csv'].bispectrum_I)
#b = np.abs(bispectra['Q_CA_8bit_1412.csv'].bispectrum_I)
#c = list(np.sum(a - b, axis=0))
#if not any(c): print("They're identical")


