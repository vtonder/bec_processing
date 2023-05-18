from mpi4py import MPI
import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import num_ch, start_indices, xy_time_offsets, pulsars, time_chunk_size
import argparse


def get_data_window(start_index, pulse_i, samples_T, int_samples_T, tot_ndp):
    start = start_index + (pulse_i * samples_T)
    end = start + int_samples_T
    chunk_start = int(np.floor(start / time_chunk_size) * time_chunk_size)
    chunk_stop = int(np.ceil(end / time_chunk_size) * time_chunk_size)

    if chunk_stop >= tot_ndp:
        return -1, -1

    return chunk_start, chunk_stop

def get_pulse_power(dfx, chunk_start, chunk_stop, start_index, pulse_i, samples_T, int_samples_T):
    data = dfx['Data/bf_raw'][:, chunk_start:chunk_stop, :]
    pulse_start = int(start_index + (pulse_i * samples_T) - chunk_start)
    pulse_stop = pulse_start + int_samples_T
    re = data[:, pulse_start:pulse_stop, 0].astype(np.float16)
    im = data[:, pulse_start:pulse_stop, 1].astype(np.float16)

    return np.float16((re**2 + im**2) / 128**2)

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("tag", help="observation tag to process. search path: /net/com08/data6/vereese/")
args = parser.parse_args()

fx = '160464' + args.tag + '_wide_tied_array_channelised_voltage_0x.h5'
fy = '160464' + args.tag + '_wide_tied_array_channelised_voltage_0y.h5'

dfx = h5py.File('/net/com08/data6/vereese/' + fx, 'r')
dfy = h5py.File('/net/com08/data6/vereese/' + fy, 'r')

si_x = start_indices[fx] + xy_time_offsets[fx] # start index of x polarisation
si_y = start_indices[fy] + xy_time_offsets[fy]

tot_ndp_x = dfx['Data/timestamps'].shape[0] # total number of data points of x polarisation
tot_ndp_y = dfy['Data/timestamps'].shape[0]

tag = args.tag
pulsar = pulsars[tag]
samples_T = pulsar['samples_T']
int_samples_T = int(np.floor(samples_T))

ndp_x = dfx['Data/timestamps'].shape[0] - si_x # number of data points, x pol
ndp_y = dfx['Data/timestamps'].shape[0] - si_y # number of data points, y pol
if ndp_x <= ndp_y:
    ndp = ndp_x
else:
    ndp = ndp_y

num_pulses = int(np.floor(ndp / samples_T))  # number of pulses per observation
np_rank = num_pulses / size # number of pulses per rank
summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float16)

if rank == 0:
    t1 = time.time()
    print("*****INFO*****")
    print("processing            : ", pulsar['name'])
    print("start_index x pol     : ", si_x)
    print("start_index y pol     : ", si_y)
    print("total x pol data len  : ", tot_ndp_x)
    print("total y pol data len  : ", tot_ndp_y)
    print("num_data_points       : ", ndp)
    print("num_pulses            : ", num_pulses)
    print("summed_profile shape  : ", summed_profile.shape)
    print("**************")

for i in np.arange(rank*np_rank, (rank+1)*np_rank):
    chunk_start_x, chunk_stop_x = get_data_window(si_x, i, samples_T, int_samples_T, tot_ndp_x)
    chunk_start_y, chunk_stop_y = get_data_window(si_y, i, samples_T, int_samples_T, tot_ndp_y)

    if chunk_stop_x == -1 or chunk_stop_y == -1:
        break

    data_x = get_pulse_power(dfx, chunk_start_x, chunk_stop_x, si_x, i, samples_T, int_samples_T)
    data_y = get_pulse_power(dfy, chunk_start_y, chunk_stop_y, si_y, i, samples_T, int_samples_T)

    summed_profile += data_x**2 + data_y**2

if rank > 0:
    comm.Send([summed_profile, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
else:
    for i in range(1, size):
        tmp_summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float16)
        comm.Recv([tmp_summed_profile, MPI.DOUBLE], source=i, tag=15)
        summed_profile += tmp_summed_profile
    np.save('intensity' + "_" + tag, summed_profile)
    print("processing took: ", time.time() - t1)

