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

df = h5py.File('/net/com08/data6/vereese/1604641234_wide_tied_array_channelised_voltage_0x.h5', 'r', rdcc_nbytes=0)
data = df['Data/bf_raw']
start_index = start_indices['1604641234_wide_tied_array_channelised_voltage_0x.h5']

num_data_points = df['Data/timestamps'].shape[0] - start_index
M = 1024 
num_sk = int(num_data_points / M)
num_sk_rank = num_sk // size  # number of sk per rank to process 

start = int(start_index + rank*num_sk_rank*M)
stop = int(start + num_sk_rank*M)
local_data = data[:, start:stop, :].astype(np.float)/128  # get the portion of the array to be analyzed by each rank

# SK RFI mitigation
FFT_LEN = 1024
SK = np.zeros([FFT_LEN, num_sk_rank])
SK_flags = np.zeros([FFT_LEN, num_sk_rank])

if rank == 0:
    print("start index: ", start_index)
    print("start: ", start, "stop: ", stop)
    print("num_sk: ", num_sk, "num_sk_rank", num_sk_rank)
    print("SK shape: ", np.shape(SK))

for i, idx in enumerate(np.arange(0, num_sk_rank*M, M)):
    SK[:, i] = spectral_kurtosis_cm(local_data[:, idx:idx+M, 0] + 1j*local_data[:, idx:idx+M, 1], M, FFT_LEN)

# send results to rank 0
if rank > 0:
    comm.Send([SK, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
else:
    tot_SK = np.zeros([FFT_LEN, num_sk])
    tot_SK[:,0:num_sk_rank] = SK
    for i in range(1, size):  # determine the size of the array to be received from each process
        tmp_SK = np.zeros([FFT_LEN, num_sk_rank])
        comm.Recv([tmp_SK, MPI.DOUBLE], source=i, tag=15)  # receive SK results from the process
        tot_SK[:,int(i*num_sk_rank):int((i+1)*num_sk_rank)] = tmp_SK

    np.save('mpi_sk_M1024_1234_0y', tot_SK)
    print("procesing took: ", time.time() - t1)

