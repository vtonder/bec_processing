import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import num_ch, start_indices, xy_time_offsets, pulsars, time_chunk_size
import argparse

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("tag", help="observation tag to process. search path: /net/com08/data6/vereese/")
args = parser.parse_args()

fx = '160464' + args.tag + '_wide_tied_array_channelised_voltage_0x.h5'
fy = '160464' + args.tag + '_wide_tied_array_channelised_voltage_0y.h5'

dfx = h5py.File('/net/com08/data6/vereese/' + fx, 'r')
dfy = h5py.File('/net/com08/data6/vereese/' + fy, 'r')

si_x = start_indices[fx] + xy_time_offsets[fx] # start index of x polarisation
si_y = start_indices[fy] + xy_time_offsets[fy]

tot_ndp_x = dfx['Data/timestamps'].shape[0] # total number of data points of x polarisation
tot_ndp_y = dfy['Data/timestamps'].shape[0]

tag = args.tag
pulsar = pulsars[tag]
samples_T = pulsar['samples_T']
int_samples_T = int(np.floor(samples_T))

ndp_x = dfx['Data/timestamps'].shape[0] - si_x # number of data points, x pol
ndp_y = dfx['Data/timestamps'].shape[0] - si_y # number of data points, y pol
if ndp_x <= ndp_y:
    ndp = ndp_x
else:
    ndp = ndp_y

#data = df['Data/bf_raw'][:,int(start_index):,:]

num_pulses = int(np.floor(ndp / samples_T))  # number of pulses per observation
FFT_LEN = int(num_ch)
summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float16)

print("*****INFO*****")
print("processing            : ", pulsar['name'])
print("start_index x pol     : ", si_x)
print("start_index y pol     : ", si_y)
print("total x pol data len  : ", tot_ndp_x)
print("total y pol data len  : ", tot_ndp_y)
print("num_data_points       : ", ndp)
print("num_pulses            : ", num_pulses)
print("summed_profile shape  : ", summed_profile.shape)
print("**************")

for i in np.arange(num_pulses):
    start_x = si_x + i*samples_T
    end_x = start_x + int_samples_T
    chunk_start_x = int(np.floor(start_x / time_chunk_size) * time_chunk_size)
    chunk_stop_x = int(np.ceil(end_x / time_chunk_size) * time_chunk_size)
    data_window_len_x = chunk_stop_x - chunk_start_x

    start_y = si_y + i*samples_T
    end_y = start_y + int_samples_T
    chunk_start_y = int(np.floor(start_y / time_chunk_size) * time_chunk_size)
    chunk_stop_y = int(np.ceil(end_y / time_chunk_size) * time_chunk_size)
    data_window_len_y = chunk_stop_y - chunk_start_y

    #print("pulse i         : ", i)
    #print("chunk_start     : ", chunk_start)
    #print("chunk_stop      : ", chunk_stop)
    #print("data_window_len : ", data_window_len)

    if chunk_stop_x >= tot_ndp_x:
        break

    if chunk_stop_y >= tot_ndp_y:
        break

    data_x = dfx['Data/bf_raw'][:, chunk_start_x:chunk_stop_x, :]
    data_y = dfy['Data/bf_raw'][:, chunk_start_y:chunk_stop_y, :]

    pulse_start_x = int(si_x + (i*samples_T) - chunk_start_x)
    pulse_stop_x = pulse_start_x + int_samples_T

    pulse_start_y = int(si_y + (i*samples_T) - chunk_start_y)
    pulse_stop_y = pulse_start_y + int_samples_T

    re_x = data_x[:, pulse_start_x:pulse_stop_x, 0].astype(np.float16)
    im_x = data_x[:, pulse_start_x:pulse_stop_x, 1].astype(np.float16)

    re_y = data_y[:, pulse_start_y:pulse_stop_y, 0].astype(np.float16)
    im_y = data_y[:, pulse_start_y:pulse_stop_y, 1].astype(np.float16)

    summed_profile += (np.float16((re_x ** 2 + im_x ** 2)/128**2))**2 + (np.float16((re_y ** 2 + im_y ** 2)/128**2))**2

np.save('intensity' + "_" + tag, summed_profile)
print("processing took: ", time.time() - t1)

