import h5py
import numpy as np
from hos import Bispectrum
import time
import sys
sys.path.append('..')
from constants import *
from matplotlib import pyplot as plt

# Assume the distribution of the data to be 0 mean Gaussian.
# This script uses the 3 sigma normality test in order to detect outliers
# Answers: what % of outliers are in the data set

DIRECTORY = '/home/vereese/git/phd_data/'
PLOT_DATA = True
SAVE_DATA = True

vela_y = h5py.File('/home/vereese/pulsar_data/1604641064_wide_tied_array_channelised_voltage_0y.h5', 'r')
data = vela_y['Data/bf_raw'][...]
data_re = np.transpose(data[:,11620864:,0])
data_im = np.transpose(data[:,11620864:,1])
data_len = len(data_re[0]) # Re & Im freq channels will have the same length

frequencies = np.arange(856+(freq_resolution/1e6)/2,1712+(freq_resolution/1e6)/2,freq_resolution/1e6)
reversed_frequencies = list(reversed(frequencies))
rounded_frequencies = [np.round(f) for f in reversed_frequencies]

re_outliers = []
im_outliers = []

for i in np.arange(1024):
    mean_re = np.mean(data_re[i])
    mean_im = np.mean(data_im[i])
    var_re = np.var(data_re[i])
    var_im = np.var(data_im[i])
    std_re = np.sqrt(var_re/data_len)
    std_im = np.sqrt(var_im/data_len)

    # normality test boundaries for real and imaginary components
    up_re = mean_re + 3*(std_re)
    up_im = mean_im + 3*(std_im)
    lo_re = mean_re - 3*(std_re)
    lo_im = mean_im - 3*(std_im)

    re_outliers_i = []
    im_outliers_i = []

    [re_outliers_i.append(1) for val in data_re[i] if val < lo_re or up_re < val]
    [im_outliers_i.append(1) for val in data_im[i] if val < lo_im or up_im < val]

    re_outliers.append((sum(re_outliers_i)/data_len)*100)
    im_outliers.append((sum(im_outliers_i)/data_len)*100)

    print("freq ch: ",i)
    print("% real outliers: ", re_outliers[i])
    print("% imag outliers: ", im_outliers[i])

if PLOT_DATA:
    plt.figure(0)
    plt.plot(re_outliers)
    plt.title("% real outliers per frequency channel")
    plt.grid()

    plt.figure(1)
    plt.plot(im_outliers)
    plt.title("% imag outliers per frequency channel")
    plt.grid()
    plt.show()

if SAVE_DATA:
    np.save(DIRECTORY+'real_outliers', re_outliers)
    np.save(DIRECTORY+'imag_outliers', im_outliers)
