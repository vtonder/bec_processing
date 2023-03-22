import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import num_ch, start_indices, lower_limit, upper_limit, pulsars, time_chunk_size
sys.path.append('../pulsar_processing')
import argparse

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-m", dest="M", help="Number of spectra to accumulate in SK calculation", default=512)
args = parser.parse_args()

df = h5py.File('/net/com08/data6/vereese/' + args.file, 'r')
start_index = start_indices[args.file]
tot_ndp = df['Data/timestamps'].shape[0]
num_data_points = df['Data/timestamps'].shape[0] - start_index 
#data = df['Data/bf_raw'][:,int(start_index):,:]
tag = args.file[6:10] + '_'   # add last 4 digits of observation code onto the file_name
pol = args.file[-5:-3]  # polarisation 0x or 0y

pulsar = pulsars[tag[:-1]] 
samples_T = pulsar['samples_T']
int_samples_T = int(np.floor(samples_T))

num_pulses = int(np.floor(num_data_points / samples_T))  # number of vela pulses per observation
M = int(args.M) 
num_sk_chunk = time_chunk_size/M # number of SK's in 1 chunk

sk = np.load("sk_M" + str(M) + "_" + tag + pol + ".npy")
FFT_LEN = int(num_ch) 
low = lower_limit[M] 
up = upper_limit[M] 

SK_flags = np.zeros([num_ch, int(num_data_points/M)], dtype=np.float16) # spans all the number of sk sets and indicated which ones were flagged
SK_flags2 = np.zeros([num_ch, int(num_data_points)], dtype=np.float16) # make this so that it can be folded ito num vela pulses
summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float16)
summed_flags = np.zeros([num_ch, int_samples_T], dtype=np.float16)

print("***SHAPES***")
print("sk             :", sk.shape)
print("sk_flags       :", SK_flags.shape)
print("sk_flags2      :", SK_flags2.shape)
print("summed_profile :", summed_profile.shape)
print("summed_flags   :", summed_flags.shape)
print("************")

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
    clean_data =  data[600, 0:M, 0]
    # RFI mitigation
    for j, idx in enumerate(np.arange(chunk_start, chunk_stop, M)):
        idx_start = int(idx)
        idx_stop = int(idx_start + M)
        sk_idx = int(num_sk_chunk*i + j)
        if idx_stop > tot_ndp:
            print("shortening range because otherwise it will read from memory that doesn't exist")
            print("tot_ndp : ", tot_ndp)
            print("idx_stop: ", idx_stop)
            idx_stop = tot_ndp - 1

        for ch, val in enumerate(sk[:, sk_idx]):
            if val < low or val > up:
                SK_flags[ch, sk_idx] = 1
                SK_flags2[ch, idx_start:idx_stop] = np.ones(M)
                # TODO: improve this mitigation strategy. coincidentally ch 600 works for vela and J0536
                data[ch, idx_start:idx_stop, 0] = clean_data
                data[ch, idx_start:idx_stop, 1] = clean_data

    pulse_start = int(start_index + (i*samples_T) - chunk_start)
    pulse_stop = pulse_start + int_samples_T

    #print("pulse_start      : ", pulse_start)
    #print("pulse_stop       : ", pulse_stop)
    #print("pulse stop-start : ", pulse_stop-pulse_start)

    re = np.float16(data[:, pulse_start:pulse_stop, 0]/128)
    im = np.float16(data[:, pulse_start:pulse_stop, 1]/128)
    summed_profile += re ** 2 + im ** 2
    summed_flags += SK_flags2[:, start:end]

np.save('ps_M' + str(M) + "_" + tag + pol, summed_profile)
np.save('SK_flags' + str(M) + "_" + tag + pol, SK_flags)
np.save('summed_flags' + str(M) + "_" + tag + pol, summed_flags)
print("procesing took: ", time.time() - t1)

