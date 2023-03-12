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
import argparse

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-m", dest="M", help="Number of spectra to accumulate in SK calculation", default=512)
args = parser.parse_args()

if rank == 0:
    t1 = time.time()

df = h5py.File('/net/com08/data6/vereese/' + args.file, 'r')
data = df['Data/bf_raw']
start_index = start_indices[args.file]

num_data_points = df['Data/timestamps'].shape[0] - start_index 
M = int(args.M) 
num_sk = int(num_data_points / M)
num_sk_rank = num_sk // size  # number of sk per rank to process 

start = int(start_index + rank*num_sk_rank*M)
stop = int(start + num_sk_rank*M)
local_data = data[:, start:stop, :]  # get the portion of the array to be analyzed by each rank

# SK RFI mitigation
FFT_LEN = 1024
SK = np.zeros([FFT_LEN, num_sk_rank], dtype=np.float16)

if rank == 0:
    print("start index: ", start_index)
    print("start: ", start, "stop: ", stop)
    print("num_sk: ", num_sk, "num_sk_rank", num_sk_rank)
    print("SK shape: ", np.shape(SK))

for i, idx in enumerate(np.arange(0, num_sk_rank*M, M)):
    SK[:, i] = spectral_kurtosis_cm(local_data[:, idx:idx+M, 0] + 1j*local_data[:, idx:idx+M, 1], M, FFT_LEN*2)

# send results to rank 0
if rank > 0:
    comm.Send([SK, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
else:
    tot_SK = np.zeros([FFT_LEN, num_sk])
    tot_SK[:, 0:num_sk_rank] = SK
    for i in range(1, size):  # determine the size of the array to be received from each process
        tmp_SK = np.zeros([FFT_LEN, num_sk_rank])
        comm.Recv([tmp_SK, MPI.DOUBLE], source=i, tag=15)  # receive SK results from the process
        tot_SK[:,int(i*num_sk_rank):int((i+1)*num_sk_rank)] = tmp_SK

    tag = '_' + args.file[6:10] + '_'   # add last 4 digits of observation code onto the file_name
    pol = args.file[-5:-3] # polarisation 0x or 0y

    np.save('sk_M' +str(M) + tag + pol , tot_SK)
    print("procesing took: ", time.time() - t1)

