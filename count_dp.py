import h5py
from mpi4py import MPI
import argparse
import numpy as np
from constants import start_indices, time_chunk_size, num_ch
import re
import time

"""
- A script to calculate the total number of dropped packets in a the given file
- Dropped packets are accumulated for each frequency channel
- The total number of data points is embedded into the output filename
"""

regex = re.compile(r'\d+')
# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("file", help = "observation file to process. search path: /net/com08/data6/vereese/")
args = parser.parse_args()

if rank == 0:
    print("Calculating dropped packets")
    t1 = time.time()

df = h5py.File('/net/com08/data6/vereese/' + args.file, 'r')
data = df['Data/bf_raw']
start_index = start_indices[args.file]

data_len = df['Data/timestamps'].shape[0]
num_data_points = ((data_len - start_index) // (size*time_chunk_size)) * (size*time_chunk_size)
num_data_points_rank = num_data_points / size
start = int(start_index + rank*num_data_points_rank)
stop = int(start + num_data_points_rank)
dp = np.zeros(num_ch)

for i, ld_idx in enumerate(np.arange(start, stop, time_chunk_size)):
    local_data = data[:, ld_idx:int(ld_idx+time_chunk_size), :]
    dp_ind = np.where(local_data == 0, True, False) # True where there are dropped packets
    # Only count as a dropped packet if both real and imaginary data points are 0
    dp += np.sum(np.logical_and(dp_ind[:, :, 0], dp_ind[:, :, 1]), axis=1)

if rank > 0:
    comm.Send([dp, MPI.DOUBLE], dest=0, tag=15)
else:
    for i in range(1, size):
        tmp_dp = np.zeros(num_ch)
        comm.Recv([tmp_dp, MPI.DOUBLE], source=i, tag=15)
        dp += tmp_dp

    nums_str = regex.findall(args.file)
    tag = nums_str[0][-4:] # get last 4 digits of observation file
    pol = args.file[-4]
    np.save('dp_' + str(num_data_points) + '_' + tag + '_' + pol, dp)
    print("done processing: ", time.time() - t1)