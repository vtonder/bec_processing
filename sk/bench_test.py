import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import frequencies, freq_resolution, time_resolution, num_ch, vela_samples_T, start_indices, time_chunk_size
from mpi4py import MPI
sys.path.append('../pulsar_processing')
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

if ((stop-start) % time_chunk_size):
    exit()

local_data = data[:, start:stop, :]  # get the portion of the array to be analyzed by each rank

if rank == 0:
    print("processing time", time.time()-t1)


