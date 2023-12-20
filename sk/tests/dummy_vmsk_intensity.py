from mpi4py import MPI
import h5py
import numpy as np
import time
import sys
sys.path.append("../")
from constants import num_ch, start_indices, pulsars, xy_time_offsets, time_chunk_size, sk_max_limit, upper_limit_4s, lower_limit_4s
from pulsar_processing.pulsar_functions import incoherent_dedisperse
import argparse


"""
Test voting MSK 

Script: dummy_vmsk_intensity.py
Input: dummy_sk.npy 
Output: produces dummy_VMSKX_flags_M.npy

Run: mpiexec -n 32 python dummy_vmsk_intensity.py 2210 -M256 -m 1 -n 2

Made all sk 1’s
Calculated where one should make dummy sk 0 and made these 0
Dimension: ch by time,in samples of M
These sk’s were zero’d 
(array([ 10     , 100    , 200    , 300    , 400    , 550    , 650    , 750]),
 array([53568, 53663, 53573, 53600, 53600, 53600, 53600, 53660]))

First non zero index for x pol for 2210 : 12288000
First chunk with x y offset compensated for: 26001408
M = 256
(26001408-12288000)/256
53568.0
"""

def rfi_mitigation(data, sk_flags, sk, M, data_window_len, start_index, chunk_start, pol):

    for idx in np.arange(0, data_window_len, M):
        idx_start = int(idx)
        idx_stop = int(idx_start + M)

        #sk_sum_idx = int(chunk_start+idx_start-start_index)
        #sk_idx = int(sk_sum_idx/M)
        sk_idx = int((chunk_start+idx_start-start_index)/M)
        if rank == 0:
            print("sk_idx: ", sk_idx)  
        #if sk_sum_idx < 0:
            #sk_sum_idx = 0
            #continue
        if sk_idx < 0:
            print("ERROR sk_idx:", sk_idx)
            print("first non zero index:", start_index)
            print("chunk_start:", chunk_start)
            break

        if sk_idx >= sk.shape[1]:
            print("reached end of sk_idx")
            break

        #if sk_sum_idx + M > sk_sum_flags.shape[1]:
        #    print("reached end of sk_sum_flags")
        #    break

        if idx_stop >= ndp:
            print("shortening range because otherwise it will read from memory that doesn't exist")
            print("tot_ndp : ", ndp)
            print("idx_stop: ", idx_stop)
            idx_stop = ndp - 1

        chs = np.arange(sk.shape[0])
        for ch in chs:

            ri = ch - n + 1
            rj = sk_idx - m + 1

            if ri < 0:
                ri = 0
            if rj < 0:
                rj = 0

            '''if rank == 0 and ch == 10:
                print("Before: ", pol)
                print("chunk_start: ", chunk_start, "idx_start: ", idx_start, "start_index:", start_index)
                print("sk_sum_idx: ", sk_sum_idx) 
                print("sk_idx:", sk_idx)
                print("skflags: ",sk_flags[ch, sk_idx])
                print("sk:", sk[ri:(ch+1), rj:(sk_idx+1)])'''


            if (sk[ri:(ch+1), rj:(sk_idx+1)] < low).any():
                #sk_sum_flags[ch, sk_sum_idx:sk_sum_idx+M] = np.ones(M, dtype=np.float16)
                sk_flags[ch, sk_idx] = np.uint8(1)
                # TODO: verify that indices are correct
                    
                #data[ch, idx_start:idx_stop, :] = np.random.normal(0, 14, (M, 2)) #clean_data
            #if rank == 0 and ch == 10 and sk_idx < 0 and pol == "ypol":
            #    print("after")
            #    print("sk_idx:", sk_idx)
            #    print("skflags: ",sk_flags[ch, sk_idx])
            #    print("sk:", sk[ri:(ch+1), rj:(sk_idx+1)])

    return data , sk_flags

def get_data_window(start_index, pulse_i, samples_T, int_samples_T, tot_ndp):
    start = start_index + (pulse_i * samples_T)
    end = start + int_samples_T
    chunk_start = int(np.floor(start / time_chunk_size) * time_chunk_size)
    chunk_stop = int(np.ceil(end / time_chunk_size) * time_chunk_size)

    if chunk_stop >= tot_ndp:
        return -1, -1

    return chunk_start, chunk_stop

def get_pulse_power(data, chunk_start, start_index, pulse_i, samples_T, int_samples_T): #, summed_flags):
    pulse_start = int(start_index + (pulse_i * samples_T) - chunk_start)
    pulse_stop = pulse_start + int_samples_T

    re = data[:, pulse_start:pulse_stop, 0].astype(np.float32)
    im = data[:, pulse_start:pulse_stop, 1].astype(np.float32)

    return np.float32(re**2) + np.float32(im**2) #, summed_flags[:, pulse_start:pulse_stop]

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("tag", help="observation tag to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-M", dest="M", help="Number of spectra to accumulate in SK calculation", default=512)
parser.add_argument("-m", dest="m", help="Number of time samples to add up in MSK", default=1)
parser.add_argument("-n", dest="n", help="Number of ch to add up in MSK", default=1)

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
m = int(args.m)
n = int(args.n)

low = lower_limit_4s[int(m * n * M)]
up = sk_max_limit[int(m*n*M)]#upper_limit7[int(m*n*M)]

tag = args.tag
pulsar = pulsars[tag]
samples_T = pulsar['samples_T']
int_samples_T = int(np.floor(samples_T))

skx = np.float32(np.load('dummy_sk.npy'))
sky = np.float32(np.load('dummy_sk.npy'))

ndp_x = dfx['Data/timestamps'].shape[0] - si_x # number of data points, x pol
ndp_y = dfy['Data/timestamps'].shape[0] - si_y # number of data points, y pol
if ndp_x <= ndp_y:
    ndp = ndp_x
else:
    ndp = ndp_y

num_pulses = ndp / samples_T  # number of pulses per observation
np_rank = 1 #int(np.floor(num_pulses / size)) # number of pulses per rank
summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float32)
skx_flags = np.zeros(skx.shape, dtype=np.uint8)
sky_flags = np.zeros(sky.shape, dtype=np.uint8)
#sk_sum_flags_x = np.zeros([num_ch, int(ndp_x)], dtype=np.float16)
#sk_sum_flags_y = np.zeros([num_ch, int(ndp_y)], dtype=np.float16)
#summed_flags = np.zeros([num_ch, int_samples_T], dtype=np.float16)

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
    chunk_start_x, chunk_stop_x = get_data_window(si_x, i, samples_T, int_samples_T, tot_ndp_x)
    chunk_start_y, chunk_stop_y = get_data_window(si_y, i, samples_T, int_samples_T, tot_ndp_y)
    data_len_x = chunk_stop_x - chunk_start_x
    data_len_y = chunk_stop_y - chunk_start_y

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
    #data_x, skx_flags, sk_sum_flags_x = rfi_mitigation(data_x, skx_flags, sk_sum_flags_x, skx, M, data_len_x,
    #                                                si_x, chunk_start_x)
    data_y, sky_flags = rfi_mitigation(data_y, sky_flags, sky, M, data_len_y, start_indices[fy], chunk_start_y, "ypol")
    data_x, skx_flags = rfi_mitigation(data_x, skx_flags, skx, M, data_len_x, start_indices[fx], chunk_start_x, "xpol")

    #data_y, sky_flags = rfi_mitigation(data_y, sky_flags, sky, M, data_len_y, si_y, chunk_start_y, "ypol")
    #data_x, skx_flags = rfi_mitigation(data_x, skx_flags, skx, M, data_len_x, si_x, chunk_start_x, "xpol")


    #sp_x = get_pulse_power(data_x, chunk_start_x, si_x, i, samples_T, int_samples_T)
    #sp_y = get_pulse_power(data_y, chunk_start_y, si_y, i, samples_T, int_samples_T)

    #summed_flags += np.float16(flags_x) + np.float16(flags_y)
    #summed_profile += sp_x + sp_y

if rank > 0:
    #comm.Send([summed_profile, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
    comm.Send([skx_flags, MPI.DOUBLE], dest=0, tag=17)  # send results to process 0
    comm.Send([sky_flags, MPI.DOUBLE], dest=0, tag=16)  # send results to process 0
    #comm.Send([summed_flags, MPI.DOUBLE], dest=0, tag=18)  # send results to process 0
else:

    for i in range(1, size):
        #tmp_summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float32)
        tmp_skx_flags = np.zeros(skx.shape, dtype=np.uint8)
        tmp_sky_flags = np.zeros(sky.shape, dtype=np.uint8)
        #tmp_sk_sum_flags = np.zeros([num_ch, int_samples_T], dtype=np.float16)
        #comm.Recv([tmp_summed_profile, MPI.DOUBLE], source=i, tag=15)
        comm.Recv([tmp_skx_flags, MPI.DOUBLE], source=i, tag=17)
        comm.Recv([tmp_sky_flags, MPI.DOUBLE], source=i, tag=16)
        #comm.Recv([tmp_sk_sum_flags, MPI.DOUBLE], source=i, tag=18)
        #summed_profile += np.float32(tmp_summed_profile)
        skx_flags += np.uint8(tmp_skx_flags)
        sky_flags += np.uint8(tmp_sky_flags)
        #summed_flags += np.float16(tmp_sk_sum_flags)

    #summed_profile = np.float32(incoherent_dedisperse(summed_profile, tag))
    #np.save('VMSK_intensity_low_sig4_M'+ str(M) + "_m" + str(m) + "_n" + str(n) + "_" + tag, summed_profile)
    np.save('dummy_VMSKX_flags_M' + str(M) + "_m" + str(m) + "_n" + str(n) + "_" + tag, skx_flags)
    np.save('dummy_VMSKY_flags_M' + str(M) + "_m" + str(m) + "_n" + str(n) + "_" + tag, sky_flags)
    #np.save('MSK_summed_flags_sig4_M' + str(M) + "_m" + str(m) + "_n" + str(n) + "_" + tag, summed_flags)
    print("processing took: ", time.time() - t1)
