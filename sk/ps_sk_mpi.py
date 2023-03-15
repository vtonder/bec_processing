import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import start_indices, time_chunk_size
from mpi4py import MPI
sys.path.append('../pulsar_processing')
from square_accumulate import *
from kurtosis import spectral_kurtosis_cm
import argparse

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-m", dest="M", help="Number of spectra to accumulate in SK calculation", default=512)
args = parser.parse_args()

M = int(args.M)

if rank == 0:
    t1 = time.time()
    if time_chunk_size % M:
        print("not respecting the chunk! M must be divisible by time_chunk_size: ", time_chunk_size)
        exit()

df = h5py.File('/net/com08/data6/vereese/' + args.file, 'r')
data = df['Data/bf_raw']
start_index = start_indices[args.file]

data_len = df['Data/timestamps'].shape[0]
num_data_points = ((data_len - start_index) // (size*time_chunk_size)) * (size*time_chunk_size)
num_data_points_rank = num_data_points / size

num_sk_rank = int(num_data_points_rank / M)  # number of sk per rank to process

start = int(start_index + rank*num_data_points_rank)
stop = int(start + num_data_points_rank)
local_data = data[:, start:stop, :]  # get the portion of the array to be analyzed by each rank

# SK RFI mitigation
FFT_LEN = 1024
SK = np.zeros([FFT_LEN, num_sk_rank], dtype=np.float16)

if rank == 0:
    print("total data_len     :", data_len)
    print("processing only    :", num_data_points)
    print("data points rank   :", num_data_points_rank)
    print("start_index        :", start_index)
    print("start              :", start)
    print("stop               :", stop)
    print("M                  :", M)
    print("SK shape           : ", np.shape(SK))

    if num_data_points_rank % time_chunk_size:
        print("not respecting the chunk! number of data points to be processed per processor must be must be divisble by time_chunk_size: ", time_chunk_size, " remainder:", num_data_points_rank % time_chunk_size)
        exit()

for i, idx in enumerate(np.arange(0, num_sk_rank*M, M)):
    SK[:, i] = spectral_kurtosis_cm(local_data[:, idx:idx+M, 0] + 1j*local_data[:, idx:idx+M, 1], M, FFT_LEN*2)

# send results to rank 0
if rank > 0:
    comm.Send([SK, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
else:
    tot_SK = np.zeros([FFT_LEN, int(num_sk_rank*size)], dtype=np.float16)
    tot_SK[:, 0:num_sk_rank] = SK
    for i in range(1, size):  # determine the size of the array to be received from each process
        tmp_SK = np.zeros([FFT_LEN, num_sk_rank], dtype=np.float16)
        comm.Recv([tmp_SK, MPI.DOUBLE], source=i, tag=15)  # receive SK results from the process
        tot_SK[:,int(i*num_sk_rank):int((i+1)*num_sk_rank)] = tmp_SK

    tag = '_' + args.file[6:10] + '_'   # add last 4 digits of observation code onto the file_name
    pol = args.file[-5:-3] # polarisation 0x or 0y

    np.save('sk_M' +str(M) + tag + pol , tot_SK)
    print("procesing took: ", time.time() - t1)

