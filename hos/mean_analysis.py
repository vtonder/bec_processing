import h5py
import numpy as np
from hos import Bispectrum
import time
import sys
sys.path.append('..')
from constants import *
from matplotlib import pyplot as plt

# Assume the distribution of the majority of data to be 0 mean Gaussian.
# This script uses the 3 sigma normality test in order to detect outliers
# Answers: what % of outliers are in the data set
# TODO:
# Create a histogram of means 
# Calculate the mean of means and standards deviation of means. ie get mean of 10000 samples , then next 

DIRECTORY = '/home/vereese/phd_data/'
OUTLIER_TEST = False
PLOT_DATA = True 
SAVE_DATA = False 

#vela_y = h5py.File('/home/vereese/pulsar_data/1604641064_wide_tied_array_channelised_voltage_0y.h5', 'r')
vela_y = h5py.File('/home/vereese/pulsar_data/1604641234_wide_tied_array_channelised_voltage_0x.h5', 'r')
data = vela_y['Data/bf_raw'][...]
data_re = data[856,2204928:,0]
data_im = data[856,2204928:,1]
data_len = len(data_re) # Re & Im freq channels will have the same length
N = 10000 # number of samples in a set
S = int(data_len/N) # number of sets
means_re, meanis_im = [], []

for i in np.arange(S):



if OUTLIER_TEST:
    re_outliers = []
    im_outliers = []
    
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
    
    print("freq ch: ",i)
    print("% real outliers: ", re_outliers)
    print("% imag outliers: ", im_outliers)

if PLOT_DATA:
    plt.figure(0)
    plt.plot(data_re) #_outliers)
    plt.title("real data")
    plt.xlabel('samples')
    plt.grid()

    plt.figure(1)
    plt.plot(data_im)
    plt.title("imag data")
    plt.xlabel('samples')
    plt.grid()
    plt.show()

if SAVE_DATA:
    np.save(DIRECTORY+'real_outliers', re_outliers)
    np.save(DIRECTORY+'imag_outliers', im_outliers)
