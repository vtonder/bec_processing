import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import num_ch, start_indices, time_chunk_size, pulsars
sys.path.append('../pulsar_processing')
import argparse

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
args = parser.parse_args()

df = h5py.File('/net/com08/data6/vereese/' + args.file, 'r')
start_index = start_indices[args.file]
tot_ndp = df['Data/timestamps'].shape[0]
num_data_points = df['Data/timestamps'].shape[0] - start_index
tag = args.file[6:10] + '_'   # add last 4 digits of observation code onto the file_name
pol = args.file[-5:-3]  # polarisation 0x or 0y

pulsar = pulsars[tag[:-1]] 
samples_T = pulsar['samples_T']
int_samples_T = round(samples_T)
# TODO: to use samples_T or int_samples_T in this division?
num_pulses = int(num_data_points / samples_T)  # number of vela pulses per observation
summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float16)

print("start_index        : ", start_index)
print("num_data_points    : ", num_data_points)
print("num_pulses         : ", num_pulses)

for i in np.arange(num_pulses):
    start = start_index + i*samples_T
    end = start + int_samples_T
    chunk_start = int(np.floor(start / time_chunk_size) * time_chunk_size)
    chunk_stop = int(np.ceil(end / time_chunk_size) * time_chunk_size)

    print("pulse i       : ", i)
    #print("chunk_start   : ", chunk_start)
    #print("chunk_stop    : ", chunk_stop)
    #print("cstp - cstrt  : ", chunk_stop - chunk_start)

    if chunk_stop >= tot_ndp:
        break
    data = df['Data/bf_raw'][:, chunk_start:chunk_stop, :]
    pulse_start = int(start_index + (i*samples_T) - chunk_start)
    pulse_stop = pulse_start + int_samples_T

    #print("pulse_start      : ", pulse_start)
    #print("pulse_stop       : ", pulse_stop)
    #print("pulse stop-start : ", pulse_stop-pulse_start)

    re = np.float16(data[:, pulse_start:pulse_stop, 0]/128)
    im = np.float16(data[:, pulse_start:pulse_stop, 1]/128)
    summed_profile += re ** 2 + im ** 2

np.save('ps_' + tag + pol, summed_profile)
print("procesing took: ", time.time() - t1)

