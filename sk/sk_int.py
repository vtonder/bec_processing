import h5py
import numpy as np
import time
import sys
sys.path.append("../")
from constants import num_ch, start_indices, pulsars, xy_time_offsets, time_chunk_size, upper_limit, lower_limit
from pulsar_processing.pulsar_functions import incoherent_dedisperse
import argparse

def rfi_mitigation(data, sk_flags, sk_sum_flags, sk, M, data_window_len, start_index, chunk_start):
    clean_data_re =  data[600, 0:M, 0] # TODO: need an improved scheme ie generate Gaussian noise with same variance
    clean_data_im =  data[600, 0:M, 1] # TODO: need an improved scheme ie generate Gaussian noise with same variance
    for idx in np.arange(0, data_window_len, M):
        idx_start = int(idx)
        idx_stop = int(idx_start + M)

        sk_sum_idx = int(chunk_start+idx_start-start_index)
        sk_idx = int(sk_sum_idx/M)

        if sk_sum_idx < 0:
            sk_sum_idx = 0

        if sk_idx >= sk.shape[1]:
            print("reached end of sk_idx")
            break

        if sk_sum_idx + M >= sk_sum_flags.shape[1]:
            print("reached end of sk_sum_flags")
            break

        if idx_stop >= ndp:
            print("shortening range because otherwise it will read from memory that doesn't exist")
            print("tot_ndp : ", ndp)
            print("idx_stop: ", idx_stop)
            idx_stop = ndp - 1

        for ch, val in enumerate(sk[:, sk_idx]):
            if val < low or val > up:
                sk_flags[ch, sk_idx] = 1
                # TODO: verify that indices are correct
                sk_sum_flags[ch, sk_sum_idx:sk_sum_idx+M] = np.ones(M, dtype=np.float16)

                data[ch, idx_start:idx_stop, 0] = clean_data_re
                data[ch, idx_start:idx_stop, 1] = clean_data_im

    return data, sk_flags, sk_sum_flags

def get_data_window(start_index, pulse_i, samples_T, int_samples_T, tot_ndp):
    start = start_index + (pulse_i * samples_T)
    end = start + int_samples_T
    chunk_start = int(np.floor(start / time_chunk_size) * time_chunk_size)
    chunk_stop = int(np.ceil(end / time_chunk_size) * time_chunk_size)

    if chunk_stop >= tot_ndp:
        return -1, -1

    return chunk_start, chunk_stop

def get_pulse_power(data, chunk_start, start_index, pulse_i, samples_T, int_samples_T, summed_flags):
    pulse_start = int(start_index + (pulse_i * samples_T) - chunk_start)
    pulse_stop = pulse_start + int_samples_T

    re = data[:, pulse_start:pulse_stop, 0].astype(np.float32)
    im = data[:, pulse_start:pulse_stop, 1].astype(np.float32)

    return np.float32(re**2) + np.float32(im**2) , summed_flags[:, pulse_start:pulse_stop]

parser = argparse.ArgumentParser()
parser.add_argument("tag", help="observation tag to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-m", dest="M", help="Number of spectra to accumulate in SK calculation", default=512)
args = parser.parse_args()

fx = '160464' + args.tag + '_wide_tied_array_channelised_voltage_0x.h5'
fy = '160464' + args.tag + '_wide_tied_array_channelised_voltage_0y.h5'

dfx = h5py.File('/net/com08/data6/vereese/' + fx, 'r')
dfy = h5py.File('/net/com08/data6/vereese/' + fy, 'r')

si_x = start_indices[fx] + xy_time_offsets[fx] # start index of x polarisation
si_y = start_indices[fy] + xy_time_offsets[fy]

tot_ndp_x = dfx['Data/timestamps'].shape[0] # total number of data points of x polarisation
tot_ndp_y = dfy['Data/timestamps'].shape[0]

M = int(args.M)
low = lower_limit[M]
up = upper_limit[M]

tag = args.tag
pulsar = pulsars[tag]
samples_T = pulsar['samples_T']
int_samples_T = int(np.floor(samples_T))

skx = np.float32(np.load('MSK_M' + str(M) + "_" + tag + "_" + "0x.npy"))
sky = np.float32(np.load('MSK_M' + str(M) + "_" + tag + "_" + "0y.npy"))

ndp_x = dfx['Data/timestamps'].shape[0] - si_x # number of data points, x pol
ndp_y = dfy['Data/timestamps'].shape[0] - si_y # number of data points, y pol
if ndp_x <= ndp_y:
    ndp = ndp_x
else:
    ndp = ndp_y

num_pulses = ndp / samples_T  # number of pulses per observation
summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float32)
skx_flags = np.zeros(skx.shape, dtype=np.float16)
sky_flags = np.zeros(sky.shape, dtype=np.float16)
sk_sum_flags_x = np.zeros([num_ch, int(ndp_x)], dtype=np.float16)
sk_sum_flags_y = np.zeros([num_ch, int(ndp_y)], dtype=np.float16)
summed_flags = np.zeros([num_ch, int_samples_T], dtype=np.float16)

t1 = time.time()
print("*****INFO*****")
print("processing            : ", pulsar['name'])
print("start_index x pol     : ", si_x)
print("start_index y pol     : ", si_y)
print("total x pol data len  : ", tot_ndp_x)
print("total y pol data len  : ", tot_ndp_y)
print("num_data_points       : ", ndp)
print("num_data_points x pol : ", ndp_x)
print("num_data_points y pol : ", ndp_y)
print("num_pulses            : ", num_pulses)
print("summed_profile shape  : ", summed_profile.shape)
print("**************")
prev_start_x, prev_stop_x = 0, 0
prev_start_y, prev_stop_y = 0, 0

for i in np.arange(num_pulses):
    chunk_start_x, chunk_stop_x = get_data_window(si_x, i, samples_T, int_samples_T, tot_ndp_x)
    chunk_start_y, chunk_stop_y = get_data_window(si_y, i, samples_T, int_samples_T, tot_ndp_y)
    data_len_x = chunk_stop_x - chunk_start_x
    data_len_y = chunk_stop_y - chunk_start_y

    if chunk_stop_x == -1 or chunk_stop_y == -1:
        break

    # This code is specifically for J0437 who spins so fast that 1 chunk contains 3.4 pulses
    if prev_start_x != chunk_start_x or prev_stop_x != chunk_stop_x:
        data_x = dfx['Data/bf_raw'][:, chunk_start_x:chunk_stop_x, :]
        prev_start_x = chunk_start_x
        prev_stop_x = chunk_stop_x

    if prev_start_y != chunk_start_y or prev_stop_y != chunk_stop_y:
        data_y = dfy['Data/bf_raw'][:, chunk_start_y:chunk_stop_y, :]
        prev_start_y = chunk_start_y
        prev_stop_y = chunk_stop_y
    data_x, skx_flags, sk_sum_flags_x = rfi_mitigation(data_x, skx_flags, sk_sum_flags_x, skx, M, data_len_x,
                                                    si_x, chunk_start_x)
    data_y, sky_flags, sk_sum_flags_y = rfi_mitigation(data_y, sky_flags, sk_sum_flags_y, sky, M, data_len_y,
                                                    si_y, chunk_start_y)
    sp_x, flags_x = get_pulse_power(data_x, chunk_start_x, si_x, i, samples_T, int_samples_T, sk_sum_flags_x)
    sp_y, flags_y = get_pulse_power(data_y, chunk_start_y, si_y, i, samples_T, int_samples_T, sk_sum_flags_y)

    summed_flags += np.float16(flags_x) + np.float16(flags_y)
    summed_profile += np.float32(sp_x**2) + np.float32(sp_y**2)


summed_profile = np.float32(incoherent_dedisperse(summed_profile, tag))
np.save('MSK_intensity_M'+ str(M) + "_" + tag, summed_profile)
np.save('MSKX_flags_' + str(M) + "_" + tag, skx_flags)
np.save('MSKY_flags_' + str(M) + "_" + tag, sky_flags)
np.save('MSK_summed_flags' + str(M) + "_" + tag, summed_flags)
print("processing took: ", time.time() - t1)
