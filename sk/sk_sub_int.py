from mpi4py import MPI
import h5py
import numpy as np
import time
import sys
sys.path.append("../")
from constants import num_ch, start_indices, pulsars, xy_time_offsets, time_chunk_size, upper_limit_skmax, lower_limit_1s, upper_limit_4s, lower_limit_4s
from common import get_data_window, get_pulse_window, get_pulse_power
from pulsar_processing.pulsar_functions import incoherent_dedisperse
import argparse
from kurtosis import spectral_kurtosis_cm

def sk_mit(data, M, data_window_len):
    for idx in np.arange(0, data_window_len, M):
        idx_start = int(idx)
        idx_stop = int(idx_start + M)

        sk = spectral_kurtosis_cm(data[:, idx_start:idx_stop, 0] + 1j*data[:, idx_start:idx_stop, 1], M, 2048)

        if idx_stop >= ndp:
            print("shortening range because otherwise it will read from memory that doesn't exist")
            print("tot_ndp : ", ndp)
            print("idx_stop: ", idx_stop)
            idx_stop = ndp - 1

        for ch, val in enumerate(sk):
            if val <= low or val >= up:
                data[ch, idx_start:idx_stop, :] = 0 #np.random.normal(0, 14, M)

    return data

def pt_mit(data, std):
    threshold = 4 * std 
    abs_data = np.sqrt(np.sum(data**2, axis=2))
    indices = np.where(abs_data >= threshold, True, False)
    ind = np.zeros(np.shape(data), dtype='bool')
    ind[:, :, 0] = indices
    ind[:, :, 1] = indices

    data[ind] = 0 #np.random.normal(0, std, np.sum(ind))

    return data

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("tag", help="observation tag to process. search path: /net/com08/data6/vereese/")
parser.add_argument("p", help="Number of pulses per sub integration. calculated 157 for tag 2210 and 47 for tag 1569")
parser.add_argument("-r", dest = "rfi", help = "RFI mitigation to conduct. options = sk pt, default = None", default = None)
parser.add_argument("-M", dest="M", help="Number of spectra to accumulate in SK calculation", default=512)

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
#low = lower_limit_1s[M]
#up = upper_limit_skmax[M]
low = lower_limit_4s[M]
up = upper_limit_4s[M]

rfi = str(args.rfi)
tag = args.tag
pulsar = pulsars[tag]
samples_T = pulsar['samples_T']
int_samples_T = int(np.round(samples_T))

ndp_x = dfx['Data/timestamps'].shape[0] - si_x # number of data points, x pol
ndp_y = dfy['Data/timestamps'].shape[0] - si_y # number of data points, y pol
if ndp_x <= ndp_y:
    ndp = ndp_x
else:
    ndp = ndp_y

num_pulses = ndp / samples_T  # number of pulses per observation
np_rank = int(np.floor(num_pulses / size)) # number of pulses per processor
num_samples_rank = np_rank * samples_T # number of samples per rank
np_sub_int = int(args.p) # number of pulses per sub integration  

if np_rank % np_sub_int:
    print("number of pulses per sub integration must be a factor of the number of pulses per processor. Try 157 for 2210 (157 is the 4th factor of 1413=45216/32=np_rank) and 47 for 1569 (47 is the 3rd factor of 94=3008/32=np_rank when using 32 processors)")
    exit()

num_sub_int = int(np_rank / np_sub_int)  # number of sub integrations per processor
summed_profile = np.zeros([num_sub_int, num_ch, int_samples_T], dtype=np.float32)

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

for h in np.arange(num_sub_int):
    for i in np.arange(np_sub_int):
        pulse_i = rank*np_rank + (np_sub_int*h + i)
        chunk_start_x, chunk_stop_x = get_data_window(si_x, pulse_i, samples_T, int_samples_T, tot_ndp_x)
        chunk_start_y, chunk_stop_y = get_data_window(si_y, pulse_i, samples_T, int_samples_T, tot_ndp_y)
        data_len_x = chunk_stop_x - chunk_start_x
        data_len_y = chunk_stop_y - chunk_start_y

        if chunk_stop_x == -1 or chunk_stop_y == -1:
            break

        # This code is specifically for J0437 who spins so fast that 1 chunk contains 3.4 pulses
        if prev_start_x != chunk_start_x or prev_stop_x != chunk_stop_x:
            data_x = dfx['Data/bf_raw'][:, chunk_start_x:chunk_stop_x, :].astype(np.float32)
            prev_start_x = chunk_start_x
            prev_stop_x = chunk_stop_x
            if args.rfi:
                if rfi == "sk":
                    data_x = sk_mit(data_x, M, data_len_x)
                else:
                    data_x = pt_mit(data_x, 14)

        if prev_start_y != chunk_start_y or prev_stop_y != chunk_stop_y:
            data_y = dfy['Data/bf_raw'][:, chunk_start_y:chunk_stop_y, :].astype(np.float32)
            prev_start_y = chunk_start_y
            prev_stop_y = chunk_stop_y
            if args.rfi:
                if rfi == "sk":
                    data_y = sk_mit(data_y, M, data_len_y)
                else:
                    data_y = pt_mit(data_y, 14)

        pulse_start_x, pulse_stop_x = get_pulse_window(chunk_start_x, si_x, pulse_i, samples_T, int_samples_T)
        pulse_start_y, pulse_stop_y = get_pulse_window(chunk_start_y, si_y, pulse_i, samples_T, int_samples_T)
        
        if pulse_start_x < 0 or pulse_start_y < 0 or pulse_stop_x < 0 or pulse_stop_y < 0: 
            print("error. info:")
            print("pulse_start_x: %d, pulse_stop_x: %d, pulse_start_y: %d, pulse_stop_y: %d" % (pulse_start_x, pulse_stop_x, pulse_start_y, pulse_stop_y))
            print("chunk_start_x: %d, si_x: %d, pulse i: %d, samples_T: %f, int_samples_T: %d" % (chunk_start_x, si_x, pulse_i, samples_T, int_samples_T))
            print("chunk_start_y: %d, si_y: %d, pulse i: %d, samples_T: %f, int_samples_T: %d\n" % (chunk_start_y, si_y, pulse_i, samples_T, int_samples_T))
            exit()

        # single pulse (sp)
        sp_x = get_pulse_power(data_x, pulse_start_x, pulse_stop_x)
        sp_y = get_pulse_power(data_y, pulse_start_y, pulse_stop_y)

        sp = sp_x + sp_y
        summed_profile[h, :, :] += np.float32(incoherent_dedisperse(sp, tag))

if rank > 0:
    comm.Send([summed_profile, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
else:
    tot_sub_int_profile = np.zeros([size*num_sub_int, num_ch, int_samples_T], dtype=np.float32)
    tot_sub_int_profile[0:num_sub_int, :, :] = summed_profile
    for i in range(1, size):
        tmp_summed_profile = np.zeros([num_sub_int, num_ch, int_samples_T], dtype=np.float32)
        comm.Recv([tmp_summed_profile, MPI.DOUBLE], source=i, tag=15)
        tot_sub_int_profile[num_sub_int*i:num_sub_int*(i+1), :, :] = np.float32(tmp_summed_profile)

    if args.rfi:
        if rfi == "sk":
            np.save("sub_int_intensity_z_sk_l4sigu4sig_M" + str(M) + "_" + tag, tot_sub_int_profile)
            #np.save("sub_int_intensity_z_sk_l1siguskmax_M" + str(M) + "_" + tag, tot_sub_int_profile)
        else:
            np.save("sub_int_intensity_z_pt_" + tag, tot_sub_int_profile)
    else:
        np.save("sub_int_intensity_z_" + tag, tot_sub_int_profile)
    print("processing took: ", time.time() - t1)
