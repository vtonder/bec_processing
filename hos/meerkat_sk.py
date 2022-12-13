import h5py
import numpy as np
from kurtosis import spectral_kurtosis
import time
import sys
sys.path.append('..')
from constants import h1_ch, gps_l1_ch, gps_l2_ch, gal_e6_ch
from matplotlib import pyplot as plt

vela_y = h5py.File('/net/com08/data6/vereese/1604641569_wide_tied_array_channelised_voltage_0y.h5', 'r')
data_h1 = vela_y['Data/bf_raw'][h1_ch,13625088:,:].astype(np.float)
FFT_LEN = 1024
M = np.floor(len(data_h1[0]) / FFT_LEN)
N = FFT_LEN*M
SK = spectral_kurtosis(data_h1[0:N, 0]+1j*data_h1[0:N, 1], M, FFT_LEN)

plt.figure(0)
plt.axhline(1.6)
plt.axhline(0.4)
plt.plot(SK)
plt.grid()
plt.show()