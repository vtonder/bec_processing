import h5py
import numpy as np
from kurtosis import spectral_kurtosis_cm
import time
import sys
from mpi4py import MPI
sys.path.append('..')
from constants import start_indices, num_ch, time_chunk_size, adc_sample_rate, bw
from constants import h1_ch, gps_l1_ch, gps_l2_ch, gal_e6_ch, time_resolution
from matplotlib import pyplot as plt
import argparse
import os

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("number of processors: ", size)
print("processor: ", rank)

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-d", "--directory", dest="directory", help="path of directory to save data products to",
                    default="/home/vereese/phd_data/sk_analysis/")
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

print("start: ", start, "end: ", end)
print("chunks_rank: ", chunks_rank)

FFT_LEN = 1024
freqs = np.arange(bw,adc_sample_rate,bw/FFT_LEN)
SK = np.zeros([int(FFT_LEN), int(chunks_rank)])

for i, idx in enumerate(np.arange(start, end, time_chunk_size)):
    idx_start = idx
    idx_stop = idx + time_chunk_size
    print(i, idx_start, idx_stop)
    SK[:, i] = spectral_kurtosis_cm(data[:, idx_start:idx_stop, 0] + 1j * data[:, idx_start:idx_stop, 1], time_chunk_size, FFT_LEN)

if rank == 0:
    tot_SK = np.zeros([int(FFT_LEN), int(size*chunks_rank)])
    tot_SK[:,start:end] = SK

    for i in range(1, size):
        tmp_sk = np.zeros([FFT_LEN, chunks_rank], dtype='float64')
        comm.Recv([tmp_sk, MPI.DOUBLE], source=i, tag=14)
        tot_SK[:,i*chunks_rank:(i+1)*chunks_rank] = tmp_sk

    if not os.path.exists(args.directory):
        os.makedirs(args.directory, exist_ok=True)
    tag = args.file[6:10] + '_'   # add last 4 digits of observation code onto the file_name
    pol = args.file[-5:-3] + '_'  # polarisation 0x or 0y
    np.save(args.directory + tag + pol + 'sk', SK)

    plt.figure(0)
    plt.imshow(SK)
    plt.grid()
    plt.show()
else:
    comm.Send([SK, MPI.DOUBLE], dest=0, tag=14)


