from mpi4py import MPI
import h5py
import numpy as np
import time
from constants import num_ch, start_indices, xy_time_offsets, pulsars, time_chunk_size
from pulsar_processing.pulsar_functions import incoherent_dedisperse
import argparse
from common import get_data_window, get_pulse_window, get_pulse_power 

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

#ind = np.load("sk/max_pulses.npy")

tag = args.tag
pulsar = pulsars[tag]
samples_T = pulsar['samples_T']
int_samples_T = int(np.floor(samples_T))

ndp_x = dfx['Data/timestamps'].shape[0] - si_x # number of data points, x pol
ndp_y = dfy['Data/timestamps'].shape[0] - si_y # number of data points, y pol
if ndp_x <= ndp_y:
    ndp = ndp_x
else:
    ndp = ndp_y

num_pulses = ndp / samples_T  # number of pulses per observation
np_rank = int(np.floor(num_pulses / size)) # number of pulses per rank
summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float32)
num_nz = np.zeros([num_ch, int_samples_T], dtype=np.float32)

if rank == 0:
    t1 = time.time()
    print("*****INFO*****")
    print("processing            : ", pulsar['name'])
    print("start_index x pol     : ", si_x)
    print("start_index y pol     : ", si_y)
    print("total x pol data len  : ", tot_ndp_x)
    print("total y pol data len  : ", tot_ndp_y)
    print("num_data_points       : ", ndp)
    print("num_data_points x pol : ", ndp_x)
    print("num_data_points y pol : ", ndp_y)
    print("num_pulses            : ", num_pulses)
    print("num pulses per rank   : ", np_rank)
    print("summed_profile shape  : ", summed_profile.shape)
    print("**************")

prev_start_x, prev_stop_x = 0, 0
prev_start_y, prev_stop_y = 0, 0
for i in np.arange(rank*np_rank, (rank+1)*np_rank):
    # only sum non brightest pulses
    #if i in ind: 
    #    continue

    chunk_start_x, chunk_stop_x = get_data_window(si_x, i, samples_T, int_samples_T, tot_ndp_x)
    chunk_start_y, chunk_stop_y = get_data_window(si_y, i, samples_T, int_samples_T, tot_ndp_y)

    if chunk_stop_x == -1 or chunk_stop_y == -1:
        break

    # This code is specifically for J0437 who spins so fast that 1 chunk contains 3.4 pulses
    if prev_start_x != chunk_start_x or prev_stop_x != chunk_stop_x:
        data_x = dfx['Data/bf_raw'][:, chunk_start_x:chunk_stop_x, :]
        prev_start_x = chunk_start_x
        prev_stop_x = chunk_stop_x

    if prev_start_y != chunk_start_y or prev_stop_y != chunk_stop_y:
        data_y = dfy['Data/bf_raw'][:, chunk_start_y:chunk_stop_y, :]
        prev_start_y = chunk_start_y
        prev_stop_y = chunk_stop_y

    pulse_start_x, pulse_stop_x = get_pulse_window(chunk_start_x, si_x, i, samples_T, int_samples_T)
    pulse_start_y, pulse_stop_y = get_pulse_window(chunk_start_y, si_y, i, samples_T, int_samples_T)

    sp_x = get_pulse_power(data_x, pulse_start_x, pulse_stop_x)
    sp_y = get_pulse_power(data_y, pulse_start_y, pulse_stop_y)

    summed_profile += np.float32(sp_x + sp_y)
    num_nz += np.where(sp_x > 0, 1, 0) + np.where(sp_y > 0, 1, 0)

if rank > 0:
    comm.Send([summed_profile, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
    comm.Send([num_nz, MPI.DOUBLE], dest=0, tag=16)
else:
    for i in range(1, size):
        tmp_summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float32)
        tmp_num_nz = np.zeros([num_ch, int_samples_T], dtype=np.float32)

        comm.Recv([tmp_summed_profile, MPI.DOUBLE], source=i, tag=15)
        comm.Recv([tmp_num_nz, MPI.DOUBLE], source=i, tag=16)

        summed_profile += np.float32(tmp_summed_profile)
        num_nz += np.float32(tmp_num_nz)

    summed_profile = np.float32(incoherent_dedisperse(summed_profile, tag))
    np.save("intensity_" + tag + "_p" + str(np_rank*size), summed_profile)
    np.save("num_nz_" + tag + "_p" + str(np_rank*size), num_nz)
    print("processing took: ", time.time() - t1)

