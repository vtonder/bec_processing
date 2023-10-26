from mpi4py import MPI
import h5py
import numpy as np
import time
import sys
sys.path.append("../")
from constants import num_ch, start_indices, xy_time_offsets, pulsars, time_chunk_size
from pulsar_processing.pulsar_functions import incoherent_dedisperse
import argparse

def get_data_window(start_index, pulse_i, samples_T, int_samples_T, tot_ndp):
    start = start_index + (pulse_i * samples_T)
    end = start + int_samples_T
    chunk_start = int(np.floor(start / time_chunk_size) * time_chunk_size)
    chunk_stop = int(np.ceil(end / time_chunk_size) * time_chunk_size)

    if chunk_stop >= tot_ndp:
        return -1, -1

    return chunk_start, chunk_stop

def get_pulse_power(data, chunk_start, start_index, pulse_i, samples_T, int_samples_T, flags):
    pulse_start = int(start_index + (pulse_i * samples_T) - chunk_start)
    pulse_stop = pulse_start + int_samples_T

    re = data[:, pulse_start:pulse_stop, 0].astype(np.float32)
    im = data[:, pulse_start:pulse_stop, 1].astype(np.float32)
    
    pf = flags[:, pulse_start:pulse_stop].astype(np.float32)
    sp = np.float32(re**2) + np.float32(im**2)

    return sp, pf 

def var_threshold(data, std, M, flags):
    threshold = 4 * std 
    #num_t = np.shape(data)[1]
    abs_data = np.sqrt(data[:,:,0]**2 + data[:,:,1]**2) 
    indices = np.where(abs_data >= threshold, True, False)
    ind = np.zeros(np.shape(data), dtype='bool')
    ind[:, :, 0] = indices
    ind[:, :, 1] = indices

    flags[indices] = 1
    data[ind] = np.random.normal(0, std, sum(sum(sum(ind))))

    ''''for i in np.arange(num_ch):
        for j in np.arange(0, num_t, M):
            abs_data_mean = np.mean(abs_data[i, j:j+M])
            if abs_data_mean >= threshold:
                flags[i, j:j+M] = np.ones(M)
                data[i, j:j+M, 0] = np.random.normal(0, var, M)
                data[i, j:j+M, 1] = np.random.normal(0, var, M)'''

    return data, flags

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

ind = np.load("max_pulses.npy")

M = int(args.M)
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
summed_flags = np.zeros([num_ch, int_samples_T], dtype=np.float32)

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
for i in np.arange(rank*np_rank, (rank+1)*np_rank):
    # only sum brightest pulses
    #if i not in ind: 
    #    continue

    chunk_start_x, chunk_stop_x = get_data_window(si_x, i, samples_T, int_samples_T, tot_ndp_x)
    chunk_start_y, chunk_stop_y = get_data_window(si_y, i, samples_T, int_samples_T, tot_ndp_y)

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

    flags_x = np.zeros([num_ch, int(chunk_stop_x - chunk_start_x)], dtype=np.float32)
    flags_y = np.zeros([num_ch, int(chunk_stop_y - chunk_start_y)], dtype=np.float32)

    data_x, flags_x  = var_threshold(data_x, 14, M, flags_x)
    data_y, flags_y  = var_threshold(data_y, 14, M, flags_y)
    sp_x, pf_x = get_pulse_power(data_x, chunk_start_x, si_x, i, samples_T, int_samples_T, flags_x)
    sp_y, pf_y = get_pulse_power(data_y, chunk_start_y, si_y, i, samples_T, int_samples_T, flags_y)

    summed_flags += np.float32(pf_x + pf_y)
    summed_profile += sp_x + sp_y

if rank > 0:
    comm.Send([summed_profile, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
    comm.Send([summed_flags, MPI.DOUBLE], dest=0, tag=16)  # send results to process 0
else:
    for i in range(1, size):
        tmp_summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float32)
        tmp_summed_flags = np.zeros([num_ch, int_samples_T], dtype=np.float32)

        comm.Recv([tmp_summed_profile, MPI.DOUBLE], source=i, tag=15)
        comm.Recv([tmp_summed_flags, MPI.DOUBLE], source=i, tag=16)

        summed_profile += np.float32(tmp_summed_profile)
        summed_flags += np.float32(tmp_summed_flags)


    summed_profile = np.float32(incoherent_dedisperse(summed_profile, tag))
    np.save('var_threshold_4sig_intensity_' + tag, summed_profile)
    np.save('vt_4sig_summed_flags_' + tag, np.float32(summed_flags))

    print("processing took: ", time.time() - t1)

