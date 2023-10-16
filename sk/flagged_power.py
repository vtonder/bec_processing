from mpi4py import MPI
import h5py
import numpy as np
import time
import sys
sys.path.append("../")
from constants import num_ch, start_indices, xy_time_offsets, time_chunk_size, sk_max_limit, upper_limit7, lower_limit7
import argparse
from kurtosis import spectral_kurtosis_cm


def rfi_mitigation(data, M, mit_power):

    for idx in np.arange(0, time_chunk_size, M):
        idx_start = int(idx)
        idx_stop = int(idx_start + M)

        sk = spectral_kurtosis_cm(data[:, idx_start:idx_stop, 0] + 1j*data[:, idx_start:idx_stop, 1], M, 2048)

        if idx_stop >= ndp:
            print("shortening range because otherwise it will read from memory that doesn't exist")
            print("tot_ndp : ", ndp)
            print("idx_stop: ", idx_stop)
            idx_stop = ndp - 1

       
        for ch, val in enumerate(sk):
            if val < low: # or val > up:
                mit_power[ch,:] += data[ch, idx_start:idx_stop, 0]**2 + data[ch, idx_start:idx_stop, 1]**2

    return mit_power

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("tag", help="observation tag to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-m", dest="M", help="Number of spectra to accumulate in SK calculation", default=512)
args = parser.parse_args()

fx = '160464' + args.tag + '_wide_tied_array_channelised_voltage_0x.h5'
fy = '160464' + args.tag + '_wide_tied_array_channelised_voltage_0y.h5'

dfx = h5py.File('/net/com08/data6/vereese/' + fx, 'r')
dfy = h5py.File('/net/com08/data6/vereese/' + fy, 'r')

si_x = start_indices[fx] + xy_time_offsets[fx] # start index of x polarisation
si_y = start_indices[fy] + xy_time_offsets[fy]

tot_ndp_x = dfx['Data/timestamps'].shape[0] # total number of data points of x polarisation
tot_ndp_y = dfy['Data/timestamps'].shape[0]

M = int(args.M)
low = lower_limit7[M]
up = sk_max_limit[M]

tag = args.tag
ndp_x = dfx['Data/timestamps'].shape[0] - si_x # number of data points, x pol
ndp_y = dfy['Data/timestamps'].shape[0] - si_y # number of data points, y pol
if ndp_x <= ndp_y:
    ndp = ndp_x
else:
    ndp = ndp_y

chunk_rank = int(np.floor(ndp / time_chunk_size / size)) # number of chunks per rank
summed_power = np.zeros(num_ch, dtype=np.float32)
summed_mit_power = np.zeros(num_ch, dtype=np.float32)

if rank == 0:
    t1 = time.time()
    print("*****INFO*****")
    print("start_index x pol     : ", si_x)
    print("start_index y pol     : ", si_y)
    print("total x pol data len  : ", tot_ndp_x)
    print("total y pol data len  : ", tot_ndp_y)
    print("num_data_points       : ", chunk_rank*size)
    print("num_data_points x pol : ", ndp_x)
    print("num_data_points y pol : ", ndp_y)
    print("**************")

for i in np.arange(rank*chunk_rank, (rank+1)*chunk_rank):
    
    chunk_start_x = si_x + i*time_chunk_size
    chunk_start_y = si_y + i*time_chunk_size
    chunk_stop_x = chunk_start_x + time_chunk_size
    chunk_stop_y = chunk_start_y + time_chunk_size

    data_x = dfx['Data/bf_raw'][:, chunk_start_x:chunk_stop_x, :].astype(np.float32)
    data_y = dfy['Data/bf_raw'][:, chunk_start_y:chunk_stop_y, :].astype(np.float32)

    summed_mit_power_x = np.zeros([num_ch, M], dtype=np.float32)
    summed_mit_power_y = np.zeros([num_ch, M], dtype=np.float32)

    summed_mit_power_x = rfi_mitigation(data_x, M, summed_mit_power_x)
    summed_mit_power_y = rfi_mitigation(data_y, M, summed_mit_power_y)

    pf_x = data_x[:,:,0]**2 + data_x[:,:,1]**2
    pf_y = data_y[:,:,0]**2 + data_y[:,:,1]**2

    summed_power += np.float32(pf_x.sum(axis=1) + pf_y.sum(axis=1))
    summed_mit_power += summed_mit_power_x.sum(axis=1) + summed_mit_power_y.sum(axis=1)

if rank > 0:
    comm.Send([summed_power, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
    comm.Send([summed_mit_power, MPI.DOUBLE], dest=0, tag=16)  # send results to process 0
else:
    for i in range(1, size):
        tmp_summed_power = np.zeros(num_ch, dtype=np.float32)
        tmp_summed_mit_power = np.zeros(num_ch, dtype=np.float32)

        comm.Recv([tmp_summed_power, MPI.DOUBLE], source=i, tag=15)
        comm.Recv([tmp_summed_mit_power, MPI.DOUBLE], source=i, tag=16)

        summed_power += np.float32(tmp_summed_power)
        summed_mit_power += np.float32(tmp_summed_mit_power)

    np.save('summed_power_low_sig4_M'+ str(M) + "_" + tag, summed_power)
    np.save('summed_mit_power_low_sig4_M'+ str(M)  + "_" + tag, summed_mit_power)

    print("processing took: ", time.time() - t1)
