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

df = h5py.File('/net/com08/data6/vereese/1604641234_wide_tied_array_channelised_voltage_0x.h5', 'r')
data = df['Data/bf_raw']
start_index = start_indices['1604641234_wide_tied_array_channelised_voltage_0x.h5']

M = 512
num_data_points = df['Data/timestamps'].shape[0] - start_index
num_sk = int(num_data_points/M/size)

print("num_sk: ", num_sk)

start = start_index + rank*num_sk*M
stop = start + num_sk*M
#print(start, stop)
#local_data = data[:, start:stop, :].astype(np.float)  # get the portion of the array to be analyzed by each rank

# SK RFI mitigation
FFT_LEN = int(1024)
low_lim = 0.77511
up_lim = 1.3254
SK = np.zeros([FFT_LEN, num_sk])
#SK_flags = np.zeros([FFT_LEN, num_sk])

for i,idx in enumerate(np.arange(start, stop, M)):
    SK[:, i] = spectral_kurtosis_cm(data[:, idx:idx+M, 0].astype(float)/128 + 1j*data[:, idx:idx+M, 1].astype(float)/128, M, FFT_LEN)
    #for j, val in enumerate(SK[:, i]):
    #    if val < low_lim or val > up_lim:
    #        SK_flags[j, i] = 1

# send results to rank 0
if rank > 0:
    #comm.Send([SK_flags, MPI.DOUBLE], dest=0, tag=14)  # send results to process 0
    comm.Send([SK, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
else:
    tot_SK = np.zeros([FFT_LEN, size*num_sk])
    #tot_SK_flags = np.zeros([FFT_LEN, size*num_sk])
    tot_SK[:,0:num_sk] = SK
    #tot_SK_flags[:,0:num_sk] = SK_flags

    for i in range(1, size):  # determine the size of the array to be received from each process
        tmp_SK = np.zeros([FFT_LEN, num_sk])
        #tmp_SK_flags = np.zeros([FFT_LEN, num_sk])
        #comm.Recv([tmp_SK_flags, MPI.DOUBLE], source=i, tag=14)  # receive results from the process
        comm.Recv([tmp_SK, MPI.DOUBLE], source=i, tag=15)  # receive SK results from the process
        tot_SK[:,int(i*num_sk):int((i+1)*num_sk)] = tmp_SK
        #tot_SK_flags[:,int(i*num_sk):int((i+1)*num_sk)] = tmp_SK_flags

    #np.save('mpi_sk_flags_M512_1234_0x', tot_SK_flags)
    np.save('mpi_sk_M512_1234_0x', tot_SK)
    print("procesing took: ", time.time() - t1)

