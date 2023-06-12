import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import start_indices, time_chunk_size, pulsars
from mpi4py import MPI
sys.path.append('../pulsar_processing')
from kurtosis import spectral_kurtosis_cm
import argparse

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("tag", help="observation tag to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-m", dest="M", help="Number of spectra to accumulate in SK calculation", default=512)
parser.add_argument("-v", dest="var_size", help="Number of spectra to calculate variance on", default=32)
args = parser.parse_args()

fx = '160464' + args.tag + '_wide_tied_array_channelised_voltage_0x.h5'
fy = '160464' + args.tag + '_wide_tied_array_channelised_voltage_0y.h5'
M = int(args.M)
var_size = int(args.var_size)

if rank == 0:
    t1 = time.time()
    if time_chunk_size % M:
        print("not respecting the chunk! M must be divisible by time_chunk_size: ", time_chunk_size)
        exit()

dfx = h5py.File('/net/com08/data6/vereese/' + fx, 'r')
dfy = h5py.File('/net/com08/data6/vereese/' + fy, 'r')
data_x = dfx['Data/bf_raw']
data_y = dfy['Data/bf_raw']
si_x = start_indices[fx]
si_y = start_indices[fy]

ndp_x = dfx['Data/timestamps'].shape[0] - si_x # number of data points, x pol
ndp_y = dfy['Data/timestamps'].shape[0] - si_y # number of data points, y pol
if ndp_x <= ndp_y:
    ndp = ndp_x
else:
    ndp = ndp_y

num_data_points = (ndp // (size * time_chunk_size)) * (size * time_chunk_size)
num_data_points_rank = num_data_points / size

num_sk_rank = int(num_data_points_rank / M)  # number of sk per rank to process
num_sk_chunk = int(time_chunk_size / M)      # number sk in 1 chunk
num_var_rank = int(num_data_points_rank / var_size) # number of variances per rank
num_var_chunk = int(time_chunk_size / var_size)      # number variances in 1 chunk

start_x = int(si_x + rank * num_data_points_rank)
stop_x = int(start_x + num_data_points_rank)

start_y = int(si_y + rank * num_data_points_rank)
stop_y = int(start_y + num_data_points_rank)

# SK RFI mitigation
FFT_LEN = 1024
sk = np.zeros([FFT_LEN, num_sk_rank], dtype=np.float16)
vars = np.zeros([FFT_LEN, num_var_rank], dtype=np.float16)

if rank == 0:
    print("processing         :", pulsars[args.tag]['name'])
    print("number of data     :", num_data_points)
    print("data points rank   :", num_data_points_rank)
    print("number sk rank     :", num_sk_rank)
    print("number sk chunk    :", num_sk_chunk)
    print("start x            :", start_x)
    print("stop  x            :", stop_x)
    print("start y            :", start_y)
    print("stop  y            :", stop_y)
    print("M                  :", M)
    print("SK shape           : ", np.shape(sk))

    if num_data_points_rank % time_chunk_size:
        print("not respecting the chunk! number of data points to be processed per processor must be must be divisble by time_chunk_size: ", time_chunk_size, " remainder:", num_data_points_rank % time_chunk_size)
        exit()

# faster for each processor to just read and process 1 chunk at a time
for i, ndp_i in enumerate(np.arange(0, num_data_points_rank, time_chunk_size)):
    sx = start_x + ndp_i
    sy = start_y + ndp_i
    local_data_x = data_x[:, sx:sx + time_chunk_size, :].astype(np.float32)/128
    local_data_y = data_x[:, sy:sy + time_chunk_size, :].astype(np.float32)/128

    local_data = (local_data_x[:,:,0] + 1j*local_data_x[:,:,1])**2 + (local_data_y[:,:,0] + 1j*local_data_y[:,:,1])**2
    sk_offset = int(i * num_sk_chunk)
    for j, idx in enumerate(np.arange(0, time_chunk_size, M)):
        sk[:, sk_offset + j] = spectral_kurtosis_cm(local_data[:, idx:idx + M], M, FFT_LEN * 2)

    var_offset = int(i * num_var_chunk)
    for j, idx in enumerate(np.arange(0, time_chunk_size, var_size)):
        vars[:, var_offset + j] = np.var(local_data[:, idx:idx+var_size])

# send results to rank 0
if rank > 0:
    comm.Send([sk, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
    comm.Send([vars, MPI.DOUBLE], dest=0, tag=16)  # send results to process 0
else:
    tot_SK = np.zeros([FFT_LEN, int(num_sk_rank*size)], dtype=np.float16)
    tot_var = np.zeros([FFT_LEN, int(num_var_rank*size)], dtype=np.float16)
    tot_SK[:, 0:num_sk_rank] = sk
    tot_var[:,0:num_var_rank] = vars
    for i in range(1, size):  # determine the size of the array to be received from each process
        tmp_SK = np.zeros([FFT_LEN, num_sk_rank], dtype=np.float16)
        tmp_var = np.zeros([FFT_LEN, num_var_rank], dtype=np.float16)
        comm.Recv([tmp_SK, MPI.DOUBLE], source=i, tag=15)  # receive SK results from the process
        comm.Recv([tmp_var, MPI.DOUBLE], source=i, tag=16)  # receive SK results from the process
        tot_SK[:,int(i*num_sk_rank):int((i+1)*num_sk_rank)] = tmp_SK
        tot_var[:,int(i*num_var_rank):int((i+1)*num_var_rank)] = tmp_var

    np.save('sk_M' + str(M) + args.tag , tot_SK)
    np.save('var_' + str(var_size) + args.tag , tot_var)
    print("procesing took: ", time.time() - t1)

