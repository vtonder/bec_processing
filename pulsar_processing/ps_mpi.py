import h5py
import numpy as np
import time
from constants import frequencies, freq_resolution, time_resolution, num_ch, vela_samples_T
from mpi4py import MPI
from square_accumulate import *


# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

df = h5py.File('/net/com08/data6/vereese/1604641234_wide_tied_array_channelised_voltage_0x.h5', 'r', rdcc_nbytes=0)
data = df['Data/bf_raw']
num_data_points = df['Data/timestamps'].shape[0]
num_pulses = int(np.floor(num_data_points / vela_samples_T))  # number of vela pulses per observation
tot_int = int(num_pulses)
print("tot_int", tot_int, "rank", rank)
#params = np.random.random((15, 3)) * 100.0  # parameters to send to my_function
n = tot_int #params.shape[0]

count = n // size  # number of catchments for each process to analyze
remainder = n % size  # extra catchments if n is not a multiple of size

vela_int_samples_T = int(np.floor(vela_samples_T))

#if rank < remainder:  # processes with rank < remainder analyze one extra catchment
#    start = rank * (count + 1) # + 11620864  # index of first catchment to analyze
#    stop = start + (count + 1)*vela_int_samples_T  # index of last catchment to analyze
#else:
start = rank*count*vela_int_samples_T # + remainder # + 11620864
stop = start + count*vela_int_samples_T

local_data = data[:, start:stop, :].astype(np.float)  # get the portion of the array to be analyzed by each rank
local_results = np.empty((num_ch, vela_int_samples_T))
#np.empty((num_ch, vela_int_samples_T))  # create result array
#local_results[:, :local_data.shape[1]] = local_data  # write parameter values to result array
local_results = square_acc(local_data, count)  # run the function for each parameter set and rank

# send results to rank 0
if rank > 0:
    comm.Send([local_results, MPI.DOUBLE], dest=0, tag=14)  # send results to process 0
else:
    summed_profile = local_results

    #final_results = np.copy(local_results)  # initialize final results with results from process 0
    for i in range(1, size):  # determine the size of the array to be received from each process
        #    if i < remainder:
        #        rank_size = count + 1
        #    else:
        #        rank_size = count
        tmp = np.empty((num_ch, vela_int_samples_T))
        #np.empty((rank_size, final_results.shape[1]), dtype=np.float)  # create empty array to receive results
        comm.Recv([tmp, MPI.DOUBLE], source=i, tag=14)  # receive results from the process
        summed_profile += tmp #np.vstack((final_results, tmp))  # add the received results to the final results
    #print("results")
    np.save('mpi_res2', summed_profile)
