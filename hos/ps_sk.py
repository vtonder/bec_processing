import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import frequencies, freq_resolution, time_resolution, num_ch, vela_samples_T, start_indices
sys.path.append('../pulsar_processing')

t1 = time.time()

df = h5py.File('/net/com08/data6/vereese/1604641234_wide_tied_array_channelised_voltage_0y.h5', 'r', rdcc_nbytes=0)
data = df['Data/bf_raw'][...]
start_index = start_indices['1604641234_wide_tied_array_channelised_voltage_0y.h5']
num_data_points = df['Data/timestamps'].shape[0] - start_index
num_pulses = int(np.floor(num_data_points / vela_samples_T))  # number of vela pulses per observation
vela_int_samples_T = int(np.floor(vela_samples_T))

sk_flags = np.load("1234_0x_sk_flags_M512.npy")
FFT_LEN = 1024
M = 512

for i in np.arange(start_index, start_index+num_data_points, M):
    idx_start = i
    idx_stop = idx_start + M
    if idx_stop > num_data_points+start_index:
        idx_stop = num_data_points+start_index-1

    for j, val in enumerate(sk_flags[:, i]):
        if val == 1:
            data[j, idx_start:idx_stop, 0] = data[600, 0:M, 0]
            data[j, idx_start:idx_stop, 1] = data[600, 0:M, 1]

summed_profile = np.zeros([num_ch, vela_int_samples_T])
for i in np.arange(num_pulses):
    start = int(i * vela_samples_T)
    end = start + vela_int_samples_T
    if end >= num_data_points:
        break
    re = data[:, start:end, 0].astype(np.float)
    im = data[:, start:end, 1].astype(np.float)
    summed_profile += re ** 2 + im ** 2

np.save('ps_M512_1234_0y', summed_profile)
print("procesing took: ", time.time() - t1)

