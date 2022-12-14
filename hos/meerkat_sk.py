import h5py
import numpy as np
from kurtosis import spectral_kurtosis
import time
import sys
sys.path.append('..')
from constants import h1_ch, gps_l1_ch, gps_l2_ch, gal_e6_ch, time_resolution
from matplotlib import pyplot as plt

vela_y = h5py.File('/net/com08/data6/vereese/1604641569_wide_tied_array_channelised_voltage_0y.h5', 'r')
ch = h1_ch
data = vela_y['Data/bf_raw'][ch, 13625088:, :].astype(np.float)

FFT_LEN = 1024
M = np.floor(len(data[0]) / FFT_LEN)
N = FFT_LEN*M
fs = 1/time_resolution
k = np.arange(int(FFT_LEN/2))
freqs = k/FFT_LEN * fs

SK = spectral_kurtosis(data[0:N, 0] + 1j * data[0:N, 1], M, FFT_LEN)

plt.figure(0)
plt.axhline(1.6)
plt.axhline(0.4)
plt.plot(freqs, SK[0:int(FFT_LEN/2)])
plt.grid()
plt.show()