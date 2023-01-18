import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import frequencies, freq_resolution, time_resolution, num_ch, vela_samples_T, start_indices
from mpi4py import MPI
sys.path.append('../pulsar_processing')
from square_accumulate import *
from kurtosis import spectral_kurtosis_cm

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    t1 = time.time()

df = h5py.File('/net/com08/data6/vereese/1604641234_wide_tied_array_channelised_voltage_0y.h5', 'r', rdcc_nbytes=0)
data = df['Data/bf_raw']
start_index = start_indices['1604641234_wide_tied_array_channelised_voltage_0y.h5']

num_data_points = df['Data/timestamps'].shape[0] - start_index
num_pulses = int(np.floor(num_data_points / vela_samples_T))  # number of vela pulses per observation
tot_int = int(num_pulses)
print("tot_int", tot_int, "rank", rank)
n = tot_int

count = n // size  # number of catchments for each process to analyze
remainder = n % size  # extra catchments if n is not a multiple of size

vela_int_samples_T = int(np.floor(vela_samples_T))

start = start_index + rank*count*vela_int_samples_T
stop = start + count*vela_int_samples_T
local_data = data[:, start:stop, :].astype(np.float)/128  # get the portion of the array to be analyzed by each rank

# SK RFI mitigation
FFT_LEN = 1024
M = 519 
data_len = count*vela_int_samples_T
low_lim = 0.776424
up_lim = 1.32275
SK = np.zeros([int(FFT_LEN), int(data_len/M)])
SK_flags = np.zeros([int(FFT_LEN), int(data_len/M)])
for i,idx in enumerate(np.arange(0, count*vela_int_samples_T, M)):
    SK[:, i] = spectral_kurtosis_cm(local_data[:, idx:idx+M, 0] + 1j*local_data[:, idx:idx+M, 1], M, FFT_LEN)
    for j, val in enumerate(SK[:, i]):
        if val < low_lim :
            SK_flags[j, i] = 1
            local_data[j, idx:idx + M, 0] = local_data[600, 0:M, 0]# np.zeros(M)#np.random.normal(0, 1, size=M)
            local_data[j, idx:idx + M, 1] = local_data[600, 0:M, 1]# np.zeros(M)#np.random.normal(0, 1, size=M)
        #else:
        #    mean_re = np.mean(local_data[j, idx:idx + M, 0])
        #    mean_im = np.mean(local_data[j, idx:idx + M, 1])
        #          std_re = np.sqrt(np.var(local_data[j, idx:idx + M, 0]))
        #          std_im = np.sqrt(np.var(local_data[j, idx:idx + M, 1]))

# Calculate power spectrum  
local_results = np.empty((num_ch, vela_int_samples_T))
local_results = square_acc(local_data, count)  # run the function for each parameter set and rank

# send results to rank 0
if rank > 0:
    comm.Send([local_results, MPI.DOUBLE], dest=0, tag=14)  # send results to process 0
    comm.Send([SK, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
else:
    summed_profile = local_results
    tot_SK = np.zeros([int(FFT_LEN), int(data_len*size/M)])
    tot_SK[:,0:int(data_len/M)] = SK

    for i in range(1, size):  # determine the size of the array to be received from each process
        tmp = np.empty((num_ch, vela_int_samples_T))
        tmp_SK = np.zeros([int(FFT_LEN), int(data_len/M)])
        comm.Recv([tmp, MPI.DOUBLE], source=i, tag=14)  # receive results from the process
        comm.Recv([tmp_SK, MPI.DOUBLE], source=i, tag=15)  # receive SK results from the process
        summed_profile += tmp
        tot_SK[:,int(i*data_len/M):int((i+1)*data_len/M)] = tmp_SK

    np.save('mpi_ps_M519_1234_0y', summed_profile)
    np.save('mpi_sk_M519_1234_0y', tot_SK)
    print("procesing took: ", time.time() - t1)

