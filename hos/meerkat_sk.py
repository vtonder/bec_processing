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
parser.add_argument("-m", dest="M", help="Number of spectra to accumulate in SK calculation", default=512)
parser.add_argument("-l", "--lower", dest="low_lim", help="lower threshold",default=0.77511)
parser.add_argument("-u", "--upper", dest="up_lim", help="upper threshold", default=1.3254)
args = parser.parse_args()

data_file = h5py.File('/net/com08/data6/vereese/' + args.file, 'r')
data = data_file['Data/bf_raw']
start_index = start_indices[args.file]
data_len = int((data.shape[1] / time_chunk_size) * time_chunk_size - start_index)

M = args.M # must be a multiple of time_chunk_size
FFT_LEN = int(1024)
num_sk = int(data_len/(size*M))

start = int(rank * num_sk * M + start_index)
end = int(start + num_sk * M)

freqs = np.arange(bw,adc_sample_rate,bw/FFT_LEN)
SK = np.zeros([FFT_LEN, num_sk])
SK_flags = np.zeros([FFT_LEN, num_sk])

low_lim = args.low_lim
up_lim = args.up_lim

if rank == 0:
    print("start: ", start, "end: ", end)
    print("data len: ", data_len)
    print("shape SK: ", np.shape(SK))
    print(low_lim, type(low_lim))
    print(up_lim, type(up_lim))


for i, idx in enumerate(np.arange(start, end, M)):
    idx_start = idx
    idx_stop = idx + M
    SK[:, i] = spectral_kurtosis_cm(data[:, idx_start:idx_stop, 0]/128 + 1j * data[:, idx_start:idx_stop, 1]/128, M, FFT_LEN)
    for j, val in enumerate(SK[:, i]):
        if val < low_lim or val > up_lim:
            SK_flags[j, i] = 1

if rank == 0:
    idx_start = int(rank*num_sk)
    idx_end = int(idx_start + num_sk)
    tot_SK = np.zeros([FFT_LEN, int(size*num_sk)])
    tot_SK_flags = np.zeros([FFT_LEN, int(size*num_sk)])
    print("shape tot_SK: ", np.shape(tot_SK))
    tot_SK[:,idx_start:idx_end] = SK
    tot_SK_flags[:,idx_start:idx_end] = SK_flags

    for i in range(1, size):
        tmp_sk = np.zeros([FFT_LEN, num_sk], dtype='float64')
        tmp_sk_flags = np.zeros([FFT_LEN, num_sk], dtype='float64')
        comm.Recv([tmp_sk, MPI.DOUBLE], source=i, tag=14)
        comm.Recv([tmp_sk_flags, MPI.DOUBLE], source=i, tag=15)

        idx_start = int(i*num_sk)
        idx_end = int(idx_start + num_sk)

        tot_SK[:,idx_start:idx_end] = tmp_sk
        tot_SK_flags[:,idx_start:idx_end] = tmp_sk_flags

    if not os.path.exists(args.directory):
        os.makedirs(args.directory, exist_ok=True)
    tag = args.file[6:10] + '_'   # add last 4 digits of observation code onto the file_name
    pol = args.file[-5:-3] + '_'  # polarisation 0x or 0y
    np.save(args.directory + tag + pol + 'sk_M' + str(M), tot_SK)
    np.save(args.directory + tag + pol + 'sk_flags_M' + str(M), tot_SK_flags)

else:
    comm.Send([SK, MPI.DOUBLE], dest=0, tag=14)
    comm.Send([SK_flags, MPI.DOUBLE], dest=0, tag=15)


