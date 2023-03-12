import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import start_indices, time_chunk_size
from mpi4py import MPI
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
    print("start bench test at:", t1)

    if time_chunk_size % M:
        print("not respecting the chunk! M must be divisble by time_chunk_size: ", time_chunk_size) 
        exit()

df = h5py.File('/net/com08/data6/vereese/' + args.file, 'r')
data = df['Data/bf_raw']
start_index = start_indices[args.file]

data_len = df['Data/timestamps'].shape[0] 
num_data_points = ((data_len - start_index) // (size*time_chunk_size)) * (size*time_chunk_size)
num_data_points_rank = num_data_points / size

start = int(start_index + rank*num_data_points_rank)
stop = int(start + num_data_points_rank)

if rank == 0:
    print("total data_len     :", data_len)
    print("processing only    :", num_data_points)
    print("data points rank   :", num_data_points_rank)
    print("start_index        :", start_index)
    print("start              :", start)
    print("stop               :", stop)
    print("M                  :", M)

    if num_data_points_rank % time_chunk_size:
        print("not respecting the chunk! number of data points to be processed per processor must be must be divisble by time_chunk_size: ", time_chunk_size, " remainder:", num_data_points_rank % time_chunk_size)
        exit()

local_data = data[:, start:stop, :]  # get the portion of the array to be analyzed by each rank

if rank == 0:
    print("processing time    :", time.time() - t1)


