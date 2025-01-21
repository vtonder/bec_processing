from mpi4py import MPI
import h5py
import numpy as np
import time
import sys
sys.path.append("../")
from constants import num_ch, start_indices, pulsars, xy_time_offsets, dme_ch
from pulsar_processing.pulsar_functions import incoherent_dedisperse
from common import get_data_window, get_pulse_window, get_pulse_power, get_pulse_flags, get_low_limit, get_up_limit
import argparse
import scipy

# note val must either be one float or a numpy array of values
def check_low(val):
    global low
 
    return val < low

def check_up(val):
    global up

    return val > up

def check_low_up(val):
    global low
    global up

    # needs to be the bitwise | or operator, otherwise this function can't deal with both single or numpy arrays as val
    return (val < low) | (val > up)

def sk_msk(data, sk_flags, sf, sk, sk_idx, idx_start, idx_stop, check_thres_array):
    for ch, val in enumerate(sk[:, sk_idx]):
        if check_thres_array[ch](val):
            sk_flags[ch, sk_idx] = np.uint8(1)
            sf[ch, idx_start:idx_stop] = 1
            # Use to replace with noise: np.random.normal(0, 14, (M, 2)) but now 0 & normalise, to try and mimic what pulsar timing experts do
            data[ch, idx_start:idx_stop, :] = 0 

    return data, sk_flags, sf

# voting multi-scale kurtosis needs to check a range of sk values depending on m and n parameter settings
def vmsk(data, sk_flags, sf, sk, sk_idx, idx_start, idx_stop, check_thres_array):
    global m
    global n

    chs = np.arange(sk.shape[0])
    for ch in chs:
        ri = ch - n + 1
        rj = sk_idx - m + 1

        if ri < 0:
            ri = 0
        if rj < 0:
            rj = 0

        if check_thres_array[ch](sk[ri:(ch+1), rj:(sk_idx+1)]).any():
            sk_flags[ch, sk_idx] = np.uint8(1)
            sf[ch, idx_start:idx_stop] = 1
            data[ch, idx_start:idx_stop, :] = 0

    return data, sk_flags, sf

def sk_mit(data, sk_flags, sf, sk, M, data_window_len, first_non_zero_idx, chunk_start, check_thres_array, sk_type):
    """
    This function mitigates RFI in data and it is noted in sk_flags and sf

    :param data: 3d numpy array with dimensions frequency, time samples, real or imaginary format
    :param sk_flags: sk flags across observation
    :param sf: summed flags with shape == pulsar profile
    :param sk: pre-calculated SK data produced by sk.py script
    :param M: M in SK estimator. window of time samples that was used in SK calculation
    :param data_window_len: length of data window that will be mitigated. It's shape will be a multiple of time_chunk_size.
    :param first_non_zero_idx: From constants file for each pol for each set
    :param chunk_start: The index where this chunk starts
    :param check_thres_array: an array of threshold functions. this allows for applying either lower, upper, or both thresholds to each frequency channel uniquely
    :param sk_type: sk, msk, or vmsk. this is used to determine whether 1 (for sk & msk) or a range (for vmsk) of sk values needs to checked against teh threshold
    :return: data, sk_flags, sf
    """

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

        data, sk_flags, sf = sk_type(data, sk_flags, sf, sk, sk_idx, idx_start, idx_stop, check_thres_array)

    return data, sk_flags, sf

def pt_mit(data, std, sf):
    """
    Power threshold (pt) RFI mitigation technique. Use 4 sigma as threshold
    Can not have a flags matrix because it'll be the same size as the observation file which is ~500 GB
    :param data: data must be in 3d format: freq ch x time samples x re,im
    :param std: std of an observation can be obtained from mean_analysis, run mean_analysis/plot_all_var.py 
    :param sf: summed flags must be same shape as summed_profile: freq ch x pulse phase bins 
    :return:
    """

    pfa_4sigma = scipy.stats.norm.cdf(4) - scipy.stats.norm.cdf(-4)
    # this already takes the mean into account
    threshold = scipy.stats.chi2.ppf(pfa_4sigma, df = 2) * std * std
    power_data = np.sum(data**2, axis=2)
    indices = np.where(power_data >= threshold, True, False)
    ind = np.zeros(np.shape(data), dtype='bool')
    ind[:, :, 0] = indices
    ind[:, :, 1] = indices

    sf[indices] = 1
    data[ind] = 0 #np.random.normal(0, std, np.sum(ind))

    return data, sf

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("tag", help = "Observation tag to process.")
parser.add_argument("-p", dest = "path", help = "Search path: default = /net/com08/data6/vereese/", default = '/net/com08/data6/vereese/')
parser.add_argument("-r", dest = "rfi", help = "RFI mitigation to conduct. options = sk msk vmsk pt, default = None", default = None)
parser.add_argument("-M", dest = "M", help = "Number of spectra to accumulate in SK calculation. Use with -r sk or msk or vmsk.", default = 512)
parser.add_argument("-m", dest = "m", help = "Number of time samples to add up in MSK. Use with -r sk or msk or vmsk.", default = 1)
parser.add_argument("-n", dest = "n", help = "Number of ch to add up in MSK. Use with -r sk or msk or vmsk.", default = 1)
parser.add_argument("-l", dest = "low", help = "Lower threshold to use. defined in constants file. map in get_lower_limit in common.py. options: 0s, 0_5s, 1s, 2s, 2_5s, 3s, 4s, 2p, skmin", default = None)
parser.add_argument("-u", dest = "up", help = "Upper threshold to use. defined in constants file. map in get_upper_limit in common.py. options: 0s, 0_5s, 1s, 2s, 2_5s, 3s, 4s, 2p, skmax", default = None)
parser.add_argument("-d", dest = "dp", help = "How dropped packets were handled. g : replaced by Gaussian noise ; z : left as 0s. Use with -r sk or msk or vmsk.", default = "z")
parser.add_argument("-a", dest = "dme", help = "Apply lower thresholds across the band and upper thresholds only to DME freq ch's. ", default = False)
parser.add_argument("-s", dest = "std", help = "standard deviation, used as threshold with -r pt", default = 14)
args = parser.parse_args()

rfi = str(args.rfi)
dp = args.dp
tag = args.tag
pulsar = pulsars[tag]
samples_T = pulsar['samples_T']
#int_samples_T = int(np.floor(samples_T))
int_samples_T = round(samples_T)

fx = '160464' + args.tag + '_wide_tied_array_channelised_voltage_0x.h5'
fy = '160464' + args.tag + '_wide_tied_array_channelised_voltage_0y.h5'
dfx = h5py.File(args.path + fx, 'r')
dfy = h5py.File(args.path + fy, 'r')

# start indices with x, y pol time offset compensated for
si_x = start_indices[fx] + xy_time_offsets[fx]
si_y = start_indices[fy] + xy_time_offsets[fy]
# calculate number of data points (ndp)
tot_ndp_x = dfx['Data/timestamps'].shape[0]
tot_ndp_y = dfy['Data/timestamps'].shape[0]
ndp_x = tot_ndp_x - si_x # number of data points, x pol
ndp_y = tot_ndp_y - si_y # number of data points, y pol
if ndp_x <= ndp_y:
    ndp = ndp_x
else:
    ndp = ndp_y

num_pulses = ndp / samples_T  # number of pulses per observation
np_rank = int(np.floor(num_pulses / size)) # number of pulses per rank
summed_profile = np.zeros([num_ch, int_samples_T], dtype = np.float32)
summed_flags = np.zeros([num_ch, int_samples_T], dtype = np.float32)
num_nz = np.zeros([num_ch, int_samples_T], dtype = np.float32) # number of non-zero data points that went into accumulation

if rfi == "sk" or rfi == "msk" or rfi == "vmsk":
    M = int(args.M)
    m = int(args.m)
    n = int(args.n)
    if args.low and args.up:
        low, low_prefix = get_low_limit(args.low, M)
        up, up_prefix = get_up_limit(args.up, M)
        check_thres_arr = num_ch * [check_low_up]
    elif args.low:
        low, low_prefix = get_low_limit(args.low, M)
        up_prefix = ""
        check_thres_arr = num_ch * [check_low]
    elif args.up:
        up, up_prefix = get_up_limit(args.up, M)
        low_prefix = ""
        check_thres_arr = num_ch * [check_up]
    else:
        print("set limits using -l and -u")
        exit()

    # For DME experiment where lower thresholds are applied across the band and upper thresholds are only applied to DME frequency channels
    if args.dme:
        check_thres_arr = num_ch * [check_low]
        for ch in dme_ch:
            check_thres_arr[ch] = check_low_up

    if rfi == "sk":
        skx = np.float32(np.load("./sk/" + "sk_" + args.dp + "_M" + str(M) + "_m" + str(m) + "_n" + str(n) + "_" + tag + "_0x.npy"))
        sky = np.float32(np.load("./sk/" + "sk_" + args.dp + "_M" + str(M) + "_m" + str(m) + "_n" + str(n) + "_" + tag + "_0y.npy"))
    else:
        # Note, both msk and vmsk use msk data sets
        skx = np.float32(np.load("./sk/" + "msk_" + args.dp + "_M" + str(M) + "_m" + str(m) + "_n" + str(n) + "_" + tag + "_0x.npy"))
        sky = np.float32(np.load("./sk/" + "msk_" + args.dp + "_M" + str(M) + "_m" + str(m) + "_n" + str(n) + "_" + tag + "_0y.npy"))

    # The type of sk mitigation function to call
    if rfi == "vmsk":
        sk_type = vmsk 
    else:
        sk_type = sk_msk 

    x_flags = np.zeros(skx.shape, dtype=np.uint8)
    y_flags = np.zeros(sky.shape, dtype=np.uint8)

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

    # summed flags per polarisation per data_len_x(y) window
    sf_x = np.zeros([num_ch, data_len_x], dtype=np.float32)
    sf_y = np.zeros([num_ch, data_len_y], dtype=np.float32)

    # This code is specifically for J0437 who spins so fast that 1 chunk contains 3.4 pulses
    # Also, reading takes long
    if prev_start_x != chunk_start_x or prev_stop_x != chunk_stop_x:
        data_x = dfx['Data/bf_raw'][:, chunk_start_x:chunk_stop_x, :].astype(np.float32)
        prev_start_x = chunk_start_x
        prev_stop_x = chunk_stop_x
        if args.rfi == "sk" or args.rfi == "msk" or args.rfi == "vmsk":
            data_x, x_flags, sf_x = sk_mit(data_x, x_flags, sf_x, skx, M, data_len_x, start_indices[fx], chunk_start_x, check_thres_arr, sk_type)
        elif args.rfi == "pt":
            data_x, sf_x = pt_mit(data_x, int(args.std), sf_x)

    if prev_start_y != chunk_start_y or prev_stop_y != chunk_stop_y:
        data_y = dfy['Data/bf_raw'][:, chunk_start_y:chunk_stop_y, :].astype(np.float32)
        prev_start_y = chunk_start_y
        prev_stop_y = chunk_stop_y
        if args.rfi == "sk" or args.rfi == "msk" or args.rfi == "vmsk":
            data_y, sky_flags, sf_y = sk_mit(data_y, y_flags, sf_y, sky, M, data_len_y, start_indices[fy], chunk_start_y, check_thres_arr, sk_type)
        elif args.rfi == "pt":
            data_y, sf_y = pt_mit(data_y, int(args.std), sf_y)

    pulse_start_x, pulse_stop_x = get_pulse_window(chunk_start_x, si_x, i, samples_T, int_samples_T)
    pulse_start_y, pulse_stop_y = get_pulse_window(chunk_start_y, si_y, i, samples_T, int_samples_T)

    # single pulse (sp)
    sp_x = get_pulse_power(data_x, pulse_start_x, pulse_stop_x)
    sp_y = get_pulse_power(data_y, pulse_start_y, pulse_stop_y)
    summed_profile += sp_x + sp_y
    num_nz += np.where(sp_x > 0, 1, 0) + np.where(sp_y > 0, 1, 0)

    if args.rfi:
        pf_x = get_pulse_flags(sf_x, pulse_start_x, pulse_stop_x)
        pf_y = get_pulse_flags(sf_y, pulse_start_y, pulse_stop_y)
        summed_flags += pf_x + pf_y

if rank > 0:
    comm.Send([summed_profile, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
    comm.Send([num_nz, MPI.DOUBLE], dest=0, tag=16)
    if args.rfi:
        comm.Send([summed_flags, MPI.DOUBLE], dest=0, tag=17)

    if rfi == "sk" or rfi == "msk" or rfi == "vmsk":
        comm.Send([x_flags, MPI.DOUBLE], dest=0, tag=18)
        comm.Send([y_flags, MPI.DOUBLE], dest=0, tag=19)
else:
    for i in range(1, size):
        tmp_summed_profile = np.zeros([num_ch, int_samples_T], dtype=np.float32)
        tmp_num_nz = np.zeros([num_ch, int_samples_T], dtype=np.float32)
        comm.Recv([tmp_summed_profile, MPI.DOUBLE], source=i, tag=15)
        comm.Recv([tmp_num_nz, MPI.DOUBLE], source=i, tag=16)
        summed_profile += np.float32(tmp_summed_profile)
        num_nz += np.float32(tmp_num_nz)

        if args.rfi:
            tmp_summed_flags = np.zeros([num_ch, int_samples_T], dtype=np.float32)
            comm.Recv([tmp_summed_flags, MPI.DOUBLE], source=i, tag=17)
            summed_flags += np.float32(tmp_summed_flags)

        if rfi == "sk" or rfi == "msk" or rfi == "vmsk":
            tmp_skx_flags = np.zeros(skx.shape, dtype=np.uint8)
            tmp_sky_flags = np.zeros(sky.shape, dtype=np.uint8)
            comm.Recv([tmp_skx_flags, MPI.DOUBLE], source=i, tag=18)
            comm.Recv([tmp_sky_flags, MPI.DOUBLE], source=i, tag=19)

            # NOTE: Bug / feature: sk?_flags can sometimes be 2 because multiple processors can process the same chunk due to pulse being in multiple chunks
            x_flags += np.uint8(tmp_skx_flags)
            y_flags += np.uint8(tmp_sky_flags)

    summed_profile = np.float32(incoherent_dedisperse(summed_profile, tag))
    nps = str(np_rank * size) # actual number of pulses folded together
    if args.rfi:
        if args.dme:
            np.save(rfi + '_dme_intensity_' + dp + '_' + low_prefix + up_prefix + '_M'+ str(M) + '_m' + str(m) + '_n' + str(n) + '_' + tag + '_p' + nps, summed_profile)
            np.save(rfi + '_dme_summed_flags_' + dp + '_' + low_prefix + up_prefix + '_M'+ str(M) + '_m' + str(m) + '_n' + str(n) + '_' + tag + '_p' + nps, summed_flags)
            np.save(rfi + '_dme_xpol_flags_' + dp + '_' + low_prefix + up_prefix + '_M'+ str(M) + '_m' + str(m) + '_n' + str(n) + '_' + tag + '_p' + nps, x_flags)
            np.save(rfi + '_dme_ypol_flags_' + dp + '_' + low_prefix + up_prefix + '_M'+ str(M) + '_m' + str(m) + '_n' + str(n) + '_' + tag + '_p' + nps, y_flags)
            np.save(rfi + '_dme_num_nz_' + dp + '_' + low_prefix + up_prefix + '_M'+ str(M) + '_m' + str(m) + '_n' + str(n) + '_' + tag + '_p' + nps, num_nz)
        elif rfi == "sk" or rfi == "msk" or rfi == "vmsk":
            np.save(rfi + '_intensity_' + dp + '_'+ low_prefix + up_prefix + '_M'+ str(M) + '_m' + str(m) + '_n' + str(n) + '_' + tag + '_p' + nps, summed_profile)
            np.save(rfi + '_summed_flags_' + dp + '_' + low_prefix + up_prefix + '_M'+ str(M) + '_m' + str(m) + '_n' + str(n) + '_' + tag + '_p' + nps, summed_flags)
            np.save(rfi + '_xpol_flags_' + dp + '_' + low_prefix + up_prefix + '_M'+ str(M) + '_m' + str(m) + '_n' + str(n) + '_' + tag + '_p' + nps, x_flags)
            np.save(rfi + '_ypol_flags_' + dp + '_' + low_prefix + up_prefix + '_M'+ str(M) + '_m' + str(m) + '_n' + str(n) + '_' + tag + '_p' + nps, y_flags)
            np.save(rfi + '_num_nz_' + dp + '_' + low_prefix + up_prefix + '_M'+ str(M) + '_m' + str(m) + '_n' + str(n) + '_' + tag + '_p' + nps, num_nz)
        elif rfi == "pt":
            np.save(rfi + '_intensity_' + dp + '_' + tag + '_p' + nps, summed_profile)
            np.save(rfi + '_summed_flags_' + dp + '_' + tag + '_p' + nps, summed_flags)
            np.save(rfi + '_num_nz_' + dp + '_' + tag + '_p' + nps, num_nz)
    else:
        np.save('intensity_' + dp + '_'  + tag + '_p' + nps, summed_profile)
        np.save('num_nz_' + dp + '_'  + tag + '_p' + nps, num_nz)

    print("processing took: ", time.time() - t1)

