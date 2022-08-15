import h5py
import numpy as np
import time
import sys
sys.path.append('..')
from hos import Bispectrum
from constants import start_indices, num_ch
import argparse
import os

# Assume the distribution of the majority of RA data to be 0 mean Gaussian.

# Calculate the mean of a set ie get mean of N samples , then mean of next N samples
# Optionally calculate the standard error of the mean of N samples
# Optionally calculate the median of N samples
# Optionally calculate the variance of N samples
# Optionally calculate % outliers using the 3 sigma normality test
# Do analysis across all frequency channels unless 1 channel is specified using -c
# The number of sets are calculated from the amount of data in a file unless specified using -s

# Note: The std deviation of a set of means is the std error as calculated using std / sqrt(N), where std is the sample
#       standard deviation and N is the number of samples in that set
# Ref: https://www.middleprofessor.com/files/applied-biostatistics_bookdown/_book/variability-and-uncertainty-standard-deviations-standard-errors-confidence-intervals.html

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /home/vereese/pulsar_data/")
parser.add_argument("N", type=int, help="number of samples in a set")
parser.add_argument("-s", "--set", dest="set", type=int,
                    help="number of sets. If not given calculate from entire data set.")
parser.add_argument("-c", "--ch", type=int, dest="ch", help="Frequency channel to analyse. "
                                                            "If given number of channels is set to 1. "
                                                            "If omitted, analyse all channels")
parser.add_argument("-d", "--directory", dest="directory", help="path of directory to save data products to",
                    default="/home/vereese/phd_data/mean_analysis/")
parser.add_argument("-e", "--std_error", dest="std_error", action='store_true',
                    help="Compute the standard error of the mean estimate of each"
                         " sample")
parser.add_argument("-a", "--mean", dest="mean", action='store_true', help="Compute the mean of each set")
parser.add_argument("-m", "--median", dest="median", action='store_true', help="Compute the median of each set")
parser.add_argument("-v", "--variance", dest="var", action='store_true', help="Compute the variance of each set")
parser.add_argument("-o", "--outlier_detection", dest="outlier", action='store_true', help="Conduct 3 sigma outlier test")
parser.add_argument("-b", "--dc_bias_test", dest="dc_bias", action='store_true', help="Conduct DC bias test")
args = parser.parse_args()

directory = args.directory+args.file[6:10]+'/'
if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)

if args.ch:
    num_ch = 1

t1 = time.time()
data_file = h5py.File('/home/vereese/pulsar_data/'+args.file, 'r')
data = data_file['Data/bf_raw'][...]
start_index = start_indices[args.file]

if num_ch == 1:
    data_re = data[args.ch, start_index:, 0]
    data_im = data[args.ch, start_index:, 1]
    str_ch = '_1ch'+str(args.ch)+'_'
    data_len = len(data_re) # Re & Im freq channels will have the same length
    data_re = data_re.reshape(num_ch, data_len)
    data_im = data_im.reshape(num_ch, data_len)
else:
    data_re = data[:, start_index:, 0]
    data_im = data[:, start_index:, 1]
    str_ch = '_1024ch_'
    data_len = len(data_re[0, :])  # Re & Im freq channels will have the same length
print("Reading in data took: ", time.time() - t1, " s")

N = args.N  # number of samples in a set
# number of sets
if args.set:
    S = args.set
else:
    S = int(data_len / N)

pol = args.file[-5:-3] # polarisation 0x or 0y

means_re, means_im = np.zeros([num_ch, S]), np.zeros([num_ch, S])
std_err_re, std_err_im = np.zeros([num_ch, S]), np.zeros([num_ch, S])
median_re, median_im = np.zeros([num_ch, S]), np.zeros([num_ch, S])
var_re, var_im = np.zeros([num_ch, S]), np.zeros([num_ch, S])
outlier_re, outlier_im = [[] for _ in np.arange(num_ch)], [[] for _ in np.arange(num_ch)]

print("saving data to               : ", directory)
print("total data length            : ", data_len)
print("number of frequency channels : ", num_ch)
print("number of samples in a set   : ", N)
print("number of sets               : ", S)

if args.mean or args.std_error or args.median:
    t1 = time.time()
    for i in np.arange(S):
        re_part = data_re[:, i * N:(i + 1) * N]
        im_part = data_im[:, i * N:(i + 1) * N]
        if args.mean:
            means_re[:, i] = np.mean(re_part, axis=1)
            means_im[:, i] = np.mean(im_part, axis=1)
        if args.std_error:
            std_err_re[:, i] = np.sqrt(np.var(re_part, axis=1) / N)
            std_err_im[:, i] = np.sqrt(np.var(im_part, axis=1) / N)
        if args.median:
            median_re[:, i] = np.median(re_part, axis=1)
            median_im[:, i] = np.median(im_part, axis=1)
    print("Analysis took: ", time.time() - t1, " s")

# TODO: Compute outliers based on MAD rejection
# https://casper.astro.berkeley.edu/wiki/Impulsive_RFI_Excision:_CASPER_Library_Block
if args.outlier:
    t1 = time.time()
    mean_re = np.mean(data_re, axis=1)
    mean_im = np.mean(data_im, axis=1)
    var_re = np.var(data_re, axis=1)
    var_im = np.var(data_im, axis=1)
    std_re = np.sqrt(var_re)
    std_im = np.sqrt(var_im)

    up_re = mean_re + 3 * std_re
    up_im = mean_im + 3 * std_im
    lo_re = mean_re - 3 * std_re
    lo_im = mean_im - 3 * std_im

    for i in np.arange(num_ch):
        [outlier_re[i].append(1) for val in data_re[i, :] if val < lo_re[i] or up_re[i] < val]
        [outlier_im[i].append(1) for val in data_im[i, :] if val < lo_im[i] or up_im[i] < val]

    # % outliers per frequency channel
    perc_re_outliers = [(sum(ch_data) / data_len) * 100 for ch_data in outlier_re]
    perc_im_outliers = [(sum(ch_data) / data_len) * 100 for ch_data in outlier_im]

    print("Outlier detection took: ", time.time() - t1, " s")

    np.save(directory + 'real_outliers_' + pol + str_ch, perc_re_outliers)
    np.save(directory + 'imag_outliers_' + pol + str_ch, perc_im_outliers)

if args.dc_bias:
    t1 = time.time()
    num_digits = int(np.round(np.log(data_len)))
    mean_bias_re = np.zeros([num_ch, num_digits])
    mean_bias_im = np.zeros([num_ch, num_digits])

    for i in np.arange(num_digits):
        data_slice = int(data_len/(10**(num_digits-1-i)))
        mean_bias_re[:,i] = np.abs(np.mean(data_re[:,0:data_slice], axis=1))
        mean_bias_im[:,i] = np.abs(np.mean(data_im[:,0:data_slice], axis=1))

    np.save(directory + 'means_bias_re_' + pol + str_ch + str(num_digits), mean_bias_re)
    np.save(directory + 'means_bias_im_' + pol + str_ch + str(num_digits), mean_bias_im)
    print("DC bias test took", time.time()-t1)

if args.mean:
    np.save(directory + 'means_re_' + pol + str_ch + str(args.N) + '_' + str(S), means_re)
    np.save(directory + 'means_im_' + pol + str_ch + str(args.N) + '_' + str(S), means_im)
if args.std_error:
    np.save(directory + "std_err_re_" + pol + str_ch + str(args.N) + '_' + str(S), std_err_re)
    np.save(directory + "std_err_im_" + pol + str_ch + str(args.N) + '_' + str(S), std_err_im)
if args.median:
    np.save(directory + "median_re_" + pol + str_ch + str(args.N) + '_' + str(S), std_err_re)
    np.save(directory + "median_im_" + pol + str_ch + str(args.N) + '_' + str(S), std_err_im)

# TODO: delete this code.
"""print("var of sample", np.var(data_re[:,N:2*N]))
print("std of sample", np.sqrt(np.var(data_re[:,N:2*N])))
print("")

print("mean of means re", np.mean(means_re))
print("mean of means im", np.mean(means_im))
print("var of means re", np.var(means_re))
print("var of means im", np.var(means_im))
print("std dev of means re", np.sqrt(np.var(means_re)))
print("std dev of means im", np.sqrt(np.var(means_im)))
print("")

if OUTLIER_TEST:
    mean_re = np.mean(data_re)
    mean_im = np.mean(data_im)
    var_re = np.var(data_re)
    var_im = np.var(data_im)
    std_re = np.sqrt(var_re)
    std_im = np.sqrt(var_im)
    print("real mean: ", mean_re)
    print("real var: ", var_re)
    print("real std: ", std_re)
    # normality test boundaries for real and imaginary components
    up_re = mean_re + 3*(std_re)
    up_im = mean_im + 3*(std_im)
    lo_re = mean_re - 3*(std_re)
    lo_im = mean_im - 3*(std_im)
    
    print("upper limit: ", up_re,"\nlower limit: ", lo_re)
    
    re_outliers_i = []
    im_outliers_i = []
    
    [re_outliers_i.append(1) for val in data_re if val < lo_re or up_re < val]
    [im_outliers_i.append(1) for val in data_im if val < lo_im or up_im < val]
    print("number outiers: ", sum(re_outliers_i))
    print(data_len)
    print("len outliers ", len(re_outliers_i))
    re_outliers = (sum(re_outliers_i)/data_len)*100
    im_outliers = (sum(im_outliers_i)/data_len)*100

    print("% real outliers: ", re_outliers)
    print("% imag outliers: ", im_outliers)

    np.save(args.directory+'_real_outliers', re_outliers)
    np.save(args.directory+'_imag_outliers', im_outliers)"""
