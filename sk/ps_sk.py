import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import num_ch, start_indices, lower_limit, upper_limit, pulsars, time_chunk_size
sys.path.append('../pulsar_processing')
import argparse

def rfi_mitigation(data, sk_flags, sk_sum_flags, sk, pulse_i, num_sk_chunk, M, data_window_len, start_index, chunk_start):
    #print("***starting RFI mitigation***")
    # TODO: improve this mitigation strategy. coincidentally ch 600 works for vela and J0536
    clean_data =  data[600, 0:M, 0] # TODO: need an improved scheme ie generate Gaussian noise with same variance
    for j, idx in enumerate(np.arange(0, data_window_len, M)):
        idx_start = int(idx)
        idx_stop = int(idx_start + M)
        #sk_idx = int(num_sk_chunk * pulse_i + j)
        sk_sum_idx = int(chunk_start+idx_start-start_index)
        sk_idx = int(sk_sum_idx/M)

        #print("idx_start :", idx_start)
        #print("idx_stop  :", idx_stop)
        #print("sk_idx    :", sk_sum_idx/M)

        if idx_stop > tot_ndp:
            print("shortening range because otherwise it will read from memory that doesn't exist")
            print("tot_ndp : ", tot_ndp)
            print("idx_stop: ", idx_stop)
            idx_stop = tot_ndp - 1

        for ch, val in enumerate(sk[:, sk_idx]):
            if val < low or val > up:
                sk_flags[ch, sk_idx] = 1
                # TODO: verify that indices are correct
                sk_sum_flags[ch, sk_sum_idx:sk_sum_idx+M] = np.ones(M, dtype=np.float16)

                data[ch, idx_start:idx_stop, 0] = clean_data
                data[ch, idx_start:idx_stop, 1] = clean_data

    return data, sk_flags, sk_sum_flags


t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-m", dest="M", help="Number of spectra to accumulate in SK calculation. Default is None. ie no RFI mitigation", default=0)
args = parser.parse_args()

M = int(args.M)
df = h5py.File('/net/com08/data6/vereese/' + args.file, 'r')
start_index = start_indices[args.file]
tot_ndp = df['Data/timestamps'].shape[0]
num_sk_chunk = 0
tag = args.file[6:10] + '_'   # add last 4 digits of observation code onto the file_name
pol = args.file[-5:-3]  # polarisation 0x or 0y
pulsar = pulsars[tag[:-1]] 
samples_T = pulsar['samples_T']
int_samples_T = int(np.floor(samples_T))

if M:
    num_sk_chunk = time_chunk_size / M  # number of SK's in 1 chunk
    sk = np.load("sk_M" + str(M) + "_" + tag + pol + ".npy")
    low = lower_limit[M]
    up = upper_limit[M]
    num_data_points = M*sk.shape[1]
    sk_flags = np.zeros(sk.shape, dtype=np.float16)
    sk_sum_flags = np.zeros([num_ch, int(num_data_points)], dtype=np.float16) 
    #sk_flags = np.zeros([num_ch, int(num_data_points / M)], dtype=np.float16) # spans all the number of sk sets and indicated which ones were flagged
    #sk_sum_flags = np.zeros([num_ch, int(num_data_points)], dtype=np.float16) # make this so that it can be folded ito num vela pulses
    summed_flags = np.zeros([num_ch, int_samples_T], dtype=np.float16)
else:
    num_data_points = df['Data/timestamps'].shape[0] - start_index 

#data = df['Data/bf_raw'][:,int(start_index):,:]

num_pulses = int(np.floor(num_data_points / samples_T))  # number of pulses per observation
FFT_LEN = int(num_ch)
summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float16)

print("***DEBUG INFO***")
print("processing         : ", args.file)
print("start_index        : ", start_index)
print("total data len     : ", tot_ndp)
print("num_data_points    : ", num_data_points)
print("num_pulses         : ", num_pulses)
print("***SHAPES***")
if M:
    print("sk             :", sk.shape)
    print("sk_flags       :", sk_flags.shape)
    print("sk_flags2      :", sk_sum_flags.shape)
    print("summed_flags   :", summed_flags.shape)
print("summed_profile :", summed_profile.shape)
print("************")

for i in np.arange(num_pulses):
    start = start_index + i*samples_T
    end = start + int_samples_T
    chunk_start = int(np.floor(start / time_chunk_size) * time_chunk_size)
    chunk_stop = int(np.ceil(end / time_chunk_size) * time_chunk_size)
    data_window_len = chunk_stop - chunk_start

    #print("pulse i         : ", i)
    #print("chunk_start     : ", chunk_start)
    #print("chunk_stop      : ", chunk_stop)
    #print("data_window_len : ", data_window_len)

    if chunk_stop >= tot_ndp:
        break
    data = df['Data/bf_raw'][:, chunk_start:chunk_stop, :]
    pulse_start = int(start_index + (i*samples_T) - chunk_start)
    pulse_stop = pulse_start + int_samples_T

    if M:
        data, sk_flags, sk_sum_flags = rfi_mitigation(data, sk_flags, sk_sum_flags, sk, i, num_sk_chunk, M, data_window_len, start_index, chunk_start)
        summed_flags += np.float16(sk_sum_flags[:, pulse_start:pulse_stop])

    re = data[:, pulse_start:pulse_stop, 0].astype(np.float16)
    im = data[:, pulse_start:pulse_stop, 1].astype(np.float16)

    summed_profile += np.float16((re ** 2 + im ** 2)/128**2)

np.save('ps_M' + str(M) + "_" + tag + pol, summed_profile)
if M:
    np.save('SK_flags' + str(M) + "_" + tag + pol, sk_flags)
    np.save('summed_flags' + str(M) + "_" + tag + pol, summed_flags)
print("processing took: ", time.time() - t1)

