from mpi4py import MPI
import h5py
import numpy as np
import time
import sys
sys.path.append("../")
from constants import num_ch, start_indices, pulsars, xy_time_offsets, time_chunk_size, upper_limit, lower_limit
from pulsar_processing.pulsar_functions import incoherent_dedisperse
import argparse
from kurtosis import spectral_kurtosis_cm

def rfi_mitigation(data, M, data_window_len):
    std_re = np.sqrt(np.var(data[600,:,0]))
    std_im = np.sqrt(np.var(data[600,:,1]))

    for idx in np.arange(0, data_window_len, M):
        idx_start = int(idx)
        idx_stop = int(idx_start + M)

        sk = spectral_kurtosis_cm(data[:, idx_start:idx_stop, 0] + 1j*data[:, idx_start:idx_stop, 1], M, 2048)

        if idx_stop >= ndp:
            print("shortening range because otherwise it will read from memory that doesn't exist")
            print("tot_ndp : ", ndp)
            print("idx_stop: ", idx_stop)
            idx_stop = ndp - 1

        for ch, val in enumerate(sk):
            if val < low or val > up:
                data[ch, idx_start:idx_stop, 0] = np.random.normal(0,std_re,1)[0]
                data[ch, idx_start:idx_stop, 1] = np.random.normal(0,std_im,1)[0]

    return data

def get_data_window(start_index, pulse_i, samples_T, int_samples_T, tot_ndp):
    start = start_index + (pulse_i * samples_T)
    end = start + int_samples_T
    chunk_start = int(np.floor(start / time_chunk_size) * time_chunk_size)
    chunk_stop = int(np.ceil(end / time_chunk_size) * time_chunk_size)

    if chunk_stop >= tot_ndp:
        return -1, -1

    return chunk_start, chunk_stop

def get_pulse_power(data, chunk_start, start_index, pulse_i, samples_T, int_samples_T):
    pulse_start = int(start_index + (pulse_i * samples_T) - chunk_start)
    pulse_stop = pulse_start + int_samples_T

    re = data[:, pulse_start:pulse_stop, 0].astype(np.float32)
    im = data[:, pulse_start:pulse_stop, 1].astype(np.float32)

    return np.float32(re**2) + np.float32(im**2)

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

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

ndp_x = dfx['Data/timestamps'].shape[0] - si_x # number of data points, x pol
ndp_y = dfy['Data/timestamps'].shape[0] - si_y # number of data points, y pol
if ndp_x <= ndp_y:
    ndp = ndp_x
else:
    ndp = ndp_y

num_pulses = ndp / samples_T  # number of pulses per observation
np_rank = int(np.floor(num_pulses / size)) # number of pulses per rank
summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float32)

if rank == 0:
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
    print("num pulses per rank   : ", np_rank)
    print("summed_profile shape  : ", summed_profile.shape)
    print("**************")

prev_start_x, prev_stop_x = 0, 0
prev_start_y, prev_stop_y = 0, 0
pulse_start = 3042
pulse_stop = 3153
pulse_w = pulse_stop - pulse_start
snrs = np.zeros(np_rank)

for i, pulse_i in enumerate(np.arange(rank * np_rank, (rank + 1) * np_rank)):
    chunk_start_x, chunk_stop_x = get_data_window(si_x, pulse_i, samples_T, int_samples_T, tot_ndp_x)
    chunk_start_y, chunk_stop_y = get_data_window(si_y, pulse_i, samples_T, int_samples_T, tot_ndp_y)
    data_len_x = chunk_stop_x - chunk_start_x
    data_len_y = chunk_stop_y - chunk_start_y

    if chunk_stop_x == -1 or chunk_stop_y == -1:
        break

    # This code is specifically for J0437 who spins so fast that 1 chunk contains 3.4 pulses
    if prev_start_x != chunk_start_x or prev_stop_x != chunk_stop_x:
        data_x = dfx['Data/bf_raw'][:, chunk_start_x:chunk_stop_x, :].astype(np.float32)
        prev_start_x = chunk_start_x
        prev_stop_x = chunk_stop_x

    if prev_start_y != chunk_start_y or prev_stop_y != chunk_stop_y:
        data_y = dfy['Data/bf_raw'][:, chunk_start_y:chunk_stop_y, :].astype(np.float32)
        prev_start_y = chunk_start_y
        prev_stop_y = chunk_stop_y

    data_x = rfi_mitigation(data_x, M, data_len_x)
    data_y = rfi_mitigation(data_y, M, data_len_y)

    sp_x = get_pulse_power(data_x, chunk_start_x, si_x, pulse_i, samples_T, int_samples_T)
    sp_y = get_pulse_power(data_y, chunk_start_y, si_y, pulse_i, samples_T, int_samples_T)
    profile = np.float32(incoherent_dedisperse( np.float32(sp_x**2) + np.float32(sp_y**2), tag))
    snr = 0
    m = np.mean(profile[0:pulse_w])
    s = np.std(profile[0:pulse_w])
    for i in np.arange(pulse_w):
        snr = snr + (profile[pulse_start + i] - m)
    snr = snr / (s * np.sqrt(pulse_w))
    snrs[i] = snr

if rank > 0:
    comm.Send([snrs, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
else:
    tot_snrs = np.zeros(np_rank*rank)
    tot_snrs[0:np_rank] = snrs
    for i in range(1, size):
        tmp_snrs = np.zeros(np_rank, dtype=np.float32)
        comm.Recv([tmp_snrs, MPI.DOUBLE], source=i, tag=15)
        tot_snrs[i*np_rank:i*np_rank + np_rank] = np.float32(tmp_snrs)

    np.save('snrs', summed_profile)
    print("processing took: ", time.time() - t1)
