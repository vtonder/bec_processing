import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import start_indices, time_chunk_size #, pulsars
from mpi4py import MPI
import argparse

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# TODO: also do a test bench test where reading in number chunks 1 pulse has. ie 63 chunks is 1 pulse
# Test 2 runs fastest ie with 1s
parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-m", dest="M", help="Number of spectra to accumulate in SK calculation", default=512)
parser.add_argument("-t", dest="T", help="test = 1: split data length by number of processors and read in all at once.test = 2: split data length by number of processors and read in 1 chunk at a timetest = 3: split data length by number of processors and read in 1/4 of chunks at a time", default=2)
args = parser.parse_args()   

test_no = int(args.T)
M = int(args.M)
if rank == 0:
    t1 = time.time()
    print("start bench test at:", t1)

    if time_chunk_size % M:
        print("not respecting the chunk! M must be divisible by time_chunk_size: ", time_chunk_size)
        exit()

df = h5py.File('/net/com08/data6/vereese/' + args.file, 'r')
data = df['Data/bf_raw']
start_index = start_indices[args.file]
#tag = args.file[6:10] # observation tag

#pulsar = pulsars[tag]
#samples_T = pulsar['samples_T']
#int_samples_T = int(np.floor(samples_T))

data_len = df['Data/timestamps'].shape[0]
num_data_points = ((data_len - start_index) // (size*time_chunk_size)) * (size*time_chunk_size)
num_data_points_rank = num_data_points / size
num_chunks_rank = num_data_points_rank / time_chunk_size # number of chunks per processor 

start = int(start_index + rank*num_data_points_rank)
stop = int(start + num_data_points_rank)

if rank == 0:
    print("processing           :", args.file)
    print("total data_len       :", data_len)
    print("processing only      :", num_data_points)
    print("data points per rank :", num_data_points_rank)
    print("# of chunks per rank :", num_chunks_rank)
    print("start_index          :", start_index)
    print("start                :", start)
    print("stop                 :", stop)
    print("M                    :", M)

    if num_data_points_rank % time_chunk_size:
        print("not respecting the chunk! number of data points to be processed per processor must be must be divisble by time_chunk_size: ", time_chunk_size, " remainder:", num_data_points_rank % time_chunk_size)
        exit()

if test_no == 1:
    local_data = data[:, start:stop, :] 

if test_no == 2:
    for idx in np.arange(start, stop, time_chunk_size):
        local_data = data[:, idx:int(idx+time_chunk_size), :]

if test_no == 3:
    for idx in np.arange(start, stop, int(num_data_points_rank/4)):
        local_data = data[:, idx:int(idx+(num_data_points_rank/4)), :]

df.close()

if rank == 0:
    print("processing time    :", time.time() - t1)

