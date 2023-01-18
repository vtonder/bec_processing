import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import num_ch, vela_samples_T, start_indices, lower_limit, upper_limit
sys.path.append('../pulsar_processing')
import argparse

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-m", dest="M", help="Number of spectra to accumulate in SK calculation", default=512)
args = parser.parse_args()

df = h5py.File('/net/com08/data6/vereese/' + args.file, 'r')
start_index = start_indices[args.file]
data = df['Data/bf_raw'][:,start_index:,:]

num_data_points = df['Data/timestamps'].shape[0] - start_index
num_pulses = int(np.floor(num_data_points / vela_samples_T))  # number of vela pulses per observation
vela_int_samples_T = int(np.floor(vela_samples_T))
M = int(args.M) 
tag = args.file[6:10] + '_'   # add last 4 digits of observation code onto the file_name
pol = args.file[-5:-3]  # polarisation 0x or 0y

sk = np.load("sk_M" + str(M) + "_" + tag + pol + ".npy")
FFT_LEN = int(num_ch) 
low = lower_limit[M] 
up = upper_limit[M] 

for i, idx in enumerate(np.arange(0, num_data_points, M)):
    idx_start = idx
    idx_stop = idx_start + M
    if i == np.shape(sk)[1]: # or col != 0:
        break;
    if idx_stop > num_data_points+start_index:
        idx_stop = num_data_points+start_index-1

    for j, val in enumerate(sk[:, i]):
        if val < low or val > up:
            data[j, idx_start:idx_stop, 0] = data[600, 0:M, 0]
            data[j, idx_start:idx_stop, 1] = data[600, 0:M, 1]

summed_profile = np.zeros([num_ch, vela_int_samples_T])
for i in np.arange(num_pulses):
    start = int(i * vela_samples_T)
    end = start + vela_int_samples_T
    if end >= num_data_points:
        break
    re = data[:, start:end, 0].astype(np.float)/128
    im = data[:, start:end, 1].astype(np.float)/128
    summed_profile += re ** 2 + im ** 2

np.save('ps_M' + str(M) + "_" + tag + pol, summed_profile)
print("procesing took: ", time.time() - t1)

