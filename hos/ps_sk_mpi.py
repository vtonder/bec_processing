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

df = h5py.File('/net/com08/data6/vereese/1604641234_wide_tied_array_channelised_voltage_0x.h5', 'r', rdcc_nbytes=0)
data = df['Data/bf_raw']
start_index = start_indices['1604641234_wide_tied_array_channelised_voltage_0x.h5']

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

local_data = data[:, start:stop, :].astype(np.float)  # get the portion of the array to be analyzed by each rank

# SK RFI mitigation
FFT_LEN = 1024
M = 512


# Calculate power spectrum
local_results = np.empty((num_ch, vela_int_samples_T))
local_results = square_acc(local_data, count)  # run the function for each parameter set and rank

# send results to rank 0
if rank > 0:
    comm.Send([local_results, MPI.DOUBLE], dest=0, tag=14)  # send results to process 0
else:
    summed_profile = local_results

    for i in range(1, size):  # determine the size of the array to be received from each process
        tmp = np.empty((num_ch, vela_int_samples_T))
        comm.Recv([tmp, MPI.DOUBLE], source=i, tag=14)  # receive results from the process
        summed_profile += tmp

    np.save('mpi_res2', summed_profile)
