import h5py
import numpy as np
from kurtosis import spectral_kurtosis_cm
import time
import sys
from mpi4py import MPI
sys.path.append('..')
from constants import start_indices, num_ch, time_chunk_size
from constants import h1_ch, gps_l1_ch, gps_l2_ch, gal_e6_ch, time_resolution
from matplotlib import pyplot as plt

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

vela_y = h5py.File('/net/com08/data6/vereese/1604641569_wide_tied_array_channelised_voltage_0y.h5', 'r')
ch = h1_ch
data = vela_y['Data/bf_raw'][:, 831*time_chunk_size:832*time_chunk_size, :].astype(np.float)

FFT_LEN = 1024
#M = np.floor(len(data[0]) / FFT_LEN)
#N = FFT_LEN*M
fs = 1/time_resolution
k = np.arange(int(FFT_LEN/2))
freqs = k/FFT_LEN * fs

SK = spectral_kurtosis_cm(data[:, :, 0] + 1j * data[:, :, 1], time_chunk_size, FFT_LEN)

plt.figure(0)
plt.axhline(1.6)
plt.axhline(0.4)
plt.plot(freqs, SK[0:int(FFT_LEN/2)])
plt.grid()
plt.show()