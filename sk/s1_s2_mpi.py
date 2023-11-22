import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import start_indices, time_chunk_size
from common import sub_0_noise
from mpi4py import MPI
from kurtosis import s1_s2
import argparse

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("file", help = "observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-M", dest = "M", help="Number of spectra to accumulate in SK calculation", default = 512)
parser.add_argument("-s", dest = "std", help="Replace dropped packets with Gaussian noise with std set using this parameter", default = None)

args = parser.parse_args()
M = int(args.M)

if rank == 0:
    t1 = time.time()
    if time_chunk_size % M:
        print("not respecting the chunk! time_chunk_size must be divisible by M. time_chunk_size: ", time_chunk_size)
        exit()

df = h5py.File('/net/com08/data6/vereese/' + args.file, 'r')
data = df['Data/bf_raw']
start_index = start_indices[args.file]

data_len = df['Data/timestamps'].shape[0]
num_data_points = ((data_len - start_index) // (size*time_chunk_size)) * (size*time_chunk_size)
num_data_points_rank = num_data_points / size

num_s_rank = int(num_data_points_rank / M)  # number of S1's and S2's per rank to process
num_s_chunk = int(time_chunk_size / M)  # number S1's and S2's in 1 chunk
start = int(start_index + rank*num_data_points_rank)
stop = int(start + num_data_points_rank)

FFT_LEN = 1024
s1 = np.zeros([FFT_LEN, num_s_rank], np.float32)
s2 = np.zeros([FFT_LEN, num_s_rank], np.float32)

if rank == 0:
    print("processing         :", args.file)
    print("total data_len     :", data_len)
    print("processing only    :", num_data_points)
    print("data points rank   :", num_data_points_rank)
    print("number S's rank    :", num_s_rank)
    print("number S's chunk   :", num_s_chunk)
    print("start_index        :", start_index)
    print("start              :", start)
    print("stop               :", stop)
    print("M                  :", M)

    if num_data_points_rank % time_chunk_size:
        print("not respecting the chunk! number of data points to be processed per processor must be must be divisble "
              "by time_chunk_size: ", time_chunk_size, " remainder:", num_data_points_rank % time_chunk_size)
        exit()

# faster for each processor to just read and process 1 chunk at a time
for i, ld_idx in enumerate(np.arange(start, stop, time_chunk_size)):
    local_data = data[:, ld_idx:int(ld_idx+time_chunk_size), :]
    if args.std:
        local_data = sub_0_noise(local_data, int(args.std))
    s_idx_offset = i * num_s_chunk
    for j, idx in enumerate(np.arange(0, time_chunk_size, M)):
        s1[:, s_idx_offset + j], s2[:, s_idx_offset + j] = s1_s2(local_data[:, idx:idx + M, 0] + 1j*local_data[:, idx:idx + M, 1], 2 * FFT_LEN)

# accumulate results
if rank > 0:
    comm.Send([s1, MPI.DOUBLE], dest=0, tag=15)  # send S1 results to process 0
    comm.Send([s2, MPI.DOUBLE], dest=0, tag=16)  # send S2 results to process 0
else:
    tot_S1 = np.zeros([FFT_LEN, int(num_s_rank*size)], np.float32)
    tot_S2 = np.zeros([FFT_LEN, int(num_s_rank*size)], np.float32)
    tot_S1[:, 0:num_s_rank] = np.float32(s1)
    tot_S2[:, 0:num_s_rank] = np.float32(s2)
    for i in range(1, size):
        tmp_S1 = np.zeros([FFT_LEN, num_s_rank], np.float32)
        tmp_S2 = np.zeros([FFT_LEN, num_s_rank], np.float32)
        comm.Recv([tmp_S1, MPI.DOUBLE], source=i, tag=15)  # receive S1 results from the process
        comm.Recv([tmp_S2, MPI.DOUBLE], source=i, tag=16)  # receive S2 results from the process
        tot_S1[:, int(i*num_s_rank):int((i+1)*num_s_rank)] = np.float32(tmp_S1)
        tot_S2[:, int(i*num_s_rank):int((i+1)*num_s_rank)] = np.float32(tmp_S2)

    tag = '_' + args.file[6:10] + '_'  # add last 4 digits of observation code onto the file_name
    pol = args.file[-5:-3]  # polarisation 0x or 0y

    if args.std:
        dp = 'g' # handled dropped packets (dp) by replacing it with Gaussian (g) noise
    else:
        dp = 'z' # leave dropped packets (dp) as zeros (z)

    np.save('S1_' + dp + '_M' + str(M) + tag + pol, tot_S1)
    np.save('S2_' + dp + '_M' + str(M) + tag + pol, tot_S2)
    print("procesing took: ", time.time() - t1)
