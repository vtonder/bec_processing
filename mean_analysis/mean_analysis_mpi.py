import h5py
import numpy as np
import time
import sys
from mpi4py import MPI
sys.path.append('..')
from constants import start_indices, num_ch, time_chunk_size
import argparse
import os

# Assume the distribution of the majority of RA data to be 0 mean Gaussian.
# Calculate the mean of a set ie get mean
# Calculate the standard error of the mean
# Calculate the variance of N samples
# Note: The std deviation of a set of means is the std error as calculated using std / sqrt(N), where std is the sample
#       standard deviation and N is the number of samples in that set
# Ref: https://www.middleprofessor.com/files/applied-biostatistics_bookdown/_book/variability-and-uncertainty-standard-deviations-standard-errors-confidence-intervals.html

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-d", "--directory", dest="directory", help="path of directory to save data products to",
                    default="/home/vereese/phd_data/mean_analysis/")
args = parser.parse_args()

data_file = h5py.File('/net/com08/data6/vereese/' + args.file, 'r', rdcc_nbytes=0)
data = data_file['Data/bf_raw']
start_index = start_indices[args.file]
# Ensure data_len is a multiple of time_chunk_size
data_len = int((data.shape[1] / time_chunk_size) * time_chunk_size - start_index)
chunks_rank = np.floor(data_len / time_chunk_size / size)  # number of chunks per rank to process, make it a round number
data_len = int(size * chunks_rank * time_chunk_size)  # ensure data_len is a multiple of time_chunk_size
start = int(rank * chunks_rank * time_chunk_size + start_index)
end = int(start + chunks_rank * time_chunk_size)

#if rank == 0:
#    print("total data length                    : ", data_len)
#    print("number of ranks                      : ", size)
#    print("number of chunks per rank to process : ", chunks_rank)
#    print("start and end for rank 0             : ", start, end)
#    print("number of frequency channels         : ", num_ch)

t1 = MPI.Wtime()

sum_squares = np.zeros([num_ch, 2], dtype='float64')
int_means = np.zeros([int(chunks_rank), num_ch, 2], dtype='float64')

for idx, i in enumerate(range(start, end, time_chunk_size)):
    d1 = data[:, i:(i + time_chunk_size), :].astype(np.float)
    int_means[idx, :, :] = np.mean(d1, axis=1)
    sum_squares += np.sum(d1 ** 2, axis=1)

means = np.mean(int_means, axis=0)

#if rank == 0:
    #print("mean analysis took: ", MPI.Wtime() - t1, " s")
    #print("shape of means: ", means.shape)

if rank == 0:
    total_mean = means
    total_sum_squares = sum_squares
    #print("dtype total sum square:", total_sum_squares.dtype)
    for i in range(1, size):
        tmp_mean = np.zeros([num_ch, 2], dtype='float64')
        tmp_ss = np.zeros([num_ch, 2], dtype='float64')

        comm.Recv([tmp_mean, MPI.DOUBLE], source=i, tag=14)
        comm.Recv([tmp_ss, MPI.DOUBLE], source=i, tag=15)

        total_mean += tmp_mean
        total_sum_squares += tmp_ss

    total_mean = total_mean / size
    var = total_sum_squares / data_len - total_mean ** 2
    std_err = np.sqrt(var / data_len)

    # strip off last 4 digits of observation code and add it onto the directory path unless the path already contains it
    if args.file[6:10] not in args.directory:
        directory = args.directory + args.file[6:10] + '/'
    else:
        directory = args.directory
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    #print("saving data to               : ", directory)

    pol = args.file[-5:-3] + '_'  # polarisation 0x or 0y

    np.save(directory + 'means_' + pol + str(num_ch), total_mean)
    np.save(directory + 'var_' + pol + str(num_ch), var)
    np.save(directory + 'std_err_' + pol + str(num_ch), std_err)
else:
    comm.Send([means, MPI.DOUBLE], dest=0, tag=14)
    comm.Send([sum_squares, MPI.DOUBLE], dest=0, tag=15)
