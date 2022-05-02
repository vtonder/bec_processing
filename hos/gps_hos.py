import numpy as np
from matplotlib import pyplot as plt
from hos import Bispectrum
import sys
from matplotlib import cm
from matplotlib.ticker import LinearLocator

DIRECTORY = '/home/vereese/phd_data/'
class SIM_GPS:
    f_l1 = 1575.42 * 10 ** 6  # Hz
    f_l2 = 1227.6 * 10 ** 6  # Hz
    spb = 6.0 / 300.0  # seconds per bit (spb) = 6 seconds per 300 bits

    L1_L2 = {
        "I_CA_D": f_l2,
        "I_PY": f_l2,
        "I_PY_D": f_l2,
        "Q_CA_D": f_l2,
        "Q_CA": f_l2,
        "Q_L2CM_Dc": f_l2
    }

    def __init__(self, files, bits):
        self.gps_data = {}
        self.bits = bits
        self.obs_time = self.bits * SIM_GPS.spb # observation time in seconds

        for file in files:
            self.gps_data.update({file:[]})
            self.iq = file[0] # This depends on file names starting with either I or Q branch

    def populate(self, direc):
        for file_name, data  in self.gps_data.items():
            with open(direc+file_name) as f:
                for line in f.readlines():
                    data.append(float(line))

    def up_convert(self):

        for file_name, data  in self.gps_data.items():
            data_len = len(data)
            t = np.arange(self.obs_time, step=self.obs_time / data_len)

            if file_name[0] == 'I':
                self.gps_data[file_name] = data*np.cos(SIM_GPS.f_l2 * t)
            else:
                self.gps_data[file_name] = data*np.sin(SIM_GPS.f_l2 * t)



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

gps = SIM_GPS(gps_file_names, 1)
gps.populate(DIRECTORY)
gps.up_convert()

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

    b = Bispectrum(data, method='direct')
    b.direct_bispectrum()
    # -4 to get rid of .csv
    np.save(DIRECTORY+fn[:-4]+'_I_bispec',b.bispectrum_I)
    #b.calc_power_spectrum()
    #b.plot_bispectrum(name=fn)
    #bispectra.update({fn:b})
    #b.plot_power_spectrum(i,fn)
    #print(np.shape(b.full_bispec))
    #plt.show()

# tested if 2 codes are the same
# tested if gps simulator consistently gives out same pattern for same config and it does
#a = np.abs(bispectra['Q_L2CM_Dc_8bit_1412.csv'].bispectrum_I)
#b = np.abs(bispectra['Q_CA_8bit_1412.csv'].bispectrum_I)
#c = list(np.sum(a - b, axis=0))
#if not any(c): print("They're identical")


