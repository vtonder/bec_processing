from mpi4py import MPI
import h5py
import numpy as np
import time
import sys
sys.path.append("../")
from constants import num_ch, start_indices, pulsars, xy_time_offsets
from pulsar_processing.pulsar_functions import incoherent_dedisperse
from common import get_data_window
import argparse
from common_sk import get_low_limit, get_up_limit
def check_low(val):
    global low

    return val < low

def check_up(val):
    global up

    return val > up

def check_low_up(val):
    global low
    global up

    return val < low or val > up

def rfi_mitigation(data, sk_flags, sk, M, data_window_len, first_non_zero_idx, chunk_start, check_thres):

    for idx in np.arange(0, data_window_len, M):
        idx_start = int(idx)
        idx_stop = int(idx_start + M)
        sk_idx = int((chunk_start + idx_start - first_non_zero_idx) / M)

        if sk_idx >= sk.shape[1]:
            print("reached end of sk_idx")
            break

        if idx_stop >= ndp:
            print("shortening range because otherwise it will read from memory that doesn't exist")
            print("tot_ndp : ", ndp)
            print("idx_stop: ", idx_stop)
            idx_stop = ndp - 1

        for ch, val in enumerate(sk[:, sk_idx]):
            if check_thres(val):
                sk_flags[ch, sk_idx] = np.uint8(1)
                data[ch, idx_start:idx_stop, :] = np.random.normal(0, 14, (M, 2)) #clean_data

    return data , sk_flags

def get_pulse_power(data, chunk_start, start_index, pulse_i, samples_T, int_samples_T):
    pulse_start = int(start_index + (pulse_i * samples_T) - chunk_start)
    pulse_stop = pulse_start + int_samples_T

    re = data[:, pulse_start:pulse_stop, 0].astype(np.float32)
    im = data[:, pulse_start:pulse_stop, 1].astype(np.float32)

    return np.float32(re**2) + np.float32(im**2)

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("tag", help="observation tag to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-M", dest="M", help="Number of spectra to accumulate in SK calculation", default=512)
parser.add_argument("-m", dest="m", help="Number of time samples to add up in MSK", default=1)
parser.add_argument("-n", dest="n", help="Number of ch to add up in MSK", default=1)
parser.add_argument("-l", dest="low", help="Key for lower threshold to use. Keys are defined constants file. Only 0 (3 sigma) and 7 (4 sigma) now supported.", default=7)
parser.add_argument("-u", dest="up", help="Key for upper threshold to use. Keys are defined constants file. Only 0 (3 sigma), 7 (4 sigma), 8 (sk max) now supported.", default=7)
parser.add_argument("-f", dest="file_prefix", help="prefix to the output files", default="msk")
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

if args.low and args.up:
    low, low_prefix = get_low_limit(int(args.low), M)
    up, up_prefix = get_up_limit(int(args.up), M)
    check_threshold = check_low_up
elif args.low:
    low, low_prefix = get_low_limit(int(args.low), M)
    up_prefix = ""
    check_threshold = check_low
elif args.up:
    up, up_prefix = get_up_limit(int(args.up), M)
    low_prefix = ""
    check_threshold = check_up
else:
    print("Give me limits")
    exit()

tag = args.tag
pulsar = pulsars[tag]
samples_T = pulsar['samples_T']
int_samples_T = int(np.floor(samples_T))

skx = np.float32(np.load(args.file_prefix + '_M' + str(M) + "_m" + str(m) + "_n" + str(n) + "_" + tag + "_0x.npy"))
sky = np.float32(np.load(args.file_prefix + '_M' + str(M) + "_m" + str(m) + "_n" + str(n) + "_" + tag + "_0y.npy"))

ndp_x = dfx['Data/timestamps'].shape[0] - si_x # number of data points, x pol
ndp_y = dfy['Data/timestamps'].shape[0] - si_y # number of data points, y pol
if ndp_x <= ndp_y:
    ndp = ndp_x
else:
    ndp = ndp_y

num_pulses = ndp / samples_T  # number of pulses per observation
np_rank = int(np.floor(num_pulses / size)) # number of pulses per rank
summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float32)
skx_flags = np.zeros(skx.shape, dtype=np.uint8)
sky_flags = np.zeros(sky.shape, dtype=np.uint8)

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
        data_x, skx_flags = rfi_mitigation(data_x, skx_flags, skx, M, data_len_x, start_indices[fx], chunk_start_x,
                                           check_threshold)
    if prev_start_y != chunk_start_y or prev_stop_y != chunk_stop_y:
        data_y = dfy['Data/bf_raw'][:, chunk_start_y:chunk_stop_y, :]
        prev_start_y = chunk_start_y
        prev_stop_y = chunk_stop_y
        data_y, sky_flags = rfi_mitigation(data_y, sky_flags, sky, M, data_len_y, start_indices[fy], chunk_start_y, check_threshold)

    sp_x = get_pulse_power(data_x, chunk_start_x, si_x, i, samples_T, int_samples_T)
    sp_y = get_pulse_power(data_y, chunk_start_y, si_y, i, samples_T, int_samples_T)

    summed_profile += sp_x + sp_y

if rank > 0:
    comm.Send([summed_profile, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
    comm.Send([skx_flags, MPI.DOUBLE], dest=0, tag=16)  # send results to process 0
    comm.Send([sky_flags, MPI.DOUBLE], dest=0, tag=17)  # send results to process 0
else:
    for i in range(1, size):
        tmp_summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float32)
        tmp_skx_flags = np.zeros(skx.shape, dtype=np.uint8)
        tmp_sky_flags = np.zeros(sky.shape, dtype=np.uint8)
        comm.Recv([tmp_summed_profile, MPI.DOUBLE], source=i, tag=15)
        comm.Recv([tmp_skx_flags, MPI.DOUBLE], source=i, tag=16)
        comm.Recv([tmp_sky_flags, MPI.DOUBLE], source=i, tag=17)
        summed_profile += np.float32(tmp_summed_profile)
        skx_flags += np.uint8(tmp_skx_flags)
        sky_flags += np.uint8(tmp_sky_flags)

    summed_profile = np.float32(incoherent_dedisperse(summed_profile, tag))
    np.save(args.file_prefix + '_intensity_' + low_prefix + up_prefix + '_M'+ str(M) + "_m" + str(m) + "_n" + str(n) + "_" + tag + "_p" + str(np_rank*size), summed_profile)
    np.save(args.file_prefix + '_xpol_flags_' + low_prefix + up_prefix + '_M'+ str(M) + "_m" + str(m) + "_n" + str(n) + "_" + tag + "_p" + str(np_rank*size), skx_flags)
    np.save(args.file_prefix + '_ypol_flags_' + low_prefix + up_prefix + '_M'+ str(M) + "_m" + str(m) + "_n" + str(n) + "_" + tag + "_p" + str(np_rank*size), sky_flags)
    print("processing took: ", time.time() - t1)
