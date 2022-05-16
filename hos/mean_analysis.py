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
# Creates a histogram of means  
# Calculate the mean of means and standards deviation of means. ie get mean of 10000 samples , then next 
# Ref: https://www.middleprofessor.com/files/applied-biostatistics_bookdown/_book/variability-and-uncertainty-standard-deviations-standard-errors-confidence-intervals.html
# Script to show that the std deviation of set of means is the std error as calculated using std / sqrt(N)
# where std is the sample standard deviation and N is the number of samples in that set

DIRECTORY = '/home/vereese/phd_data/'
OUTLIER_TEST = False
PLOT_DATA = True 
SAVE_DATA = False 

#vela_y = h5py.File('/home/vereese/pulsar_data/1604641064_wide_tied_array_channelised_voltage_0y.h5', 'r')
vela_y = h5py.File('/home/vereese/pulsar_data/1604641234_wide_tied_array_channelised_voltage_0x.h5', 'r')
data = vela_y['Data/bf_raw'][...]
data_re = data[856,13634560:,0]
data_im = data[856,13634560:,1]
data_len = len(data_re) # Re & Im freq channels will have the same length
N = 2048 #10000 # number of samples in a set
S = int(data_len/N) # number of sets
means_re, means_im = [], []
print("number of sets:", S)
for i in np.arange(S):
    means_re.append(np.mean(data_re[i*N:(i+1)*N]))
    means_im.append(np.mean(data_im[i*N:(i+1)*N]))

print("mean of means re", np.mean(means_re))
print("mean of means im", np.mean(means_im))
print("var of means re", np.var(means_re))
print("var of means im", np.var(means_im))
print("std dev of means re", np.sqrt(np.var(means_re)))
print("std dev of means im", np.sqrt(np.var(means_im)))

print("std error of means re", np.sqrt(np.var(data_re[0:N])/N))
print("std error of means im", np.sqrt(np.var(data_im[0:N])/N))
print("std error of means re", np.sqrt(np.var(data_re[N:2*N])/N))
print("std error of means im", np.sqrt(np.var(data_im[N:2*N])/N))

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

    if SAVE_DATA:
        np.save(DIRECTORY+'real_outliers', re_outliers)
        np.save(DIRECTORY+'imag_outliers', im_outliers)

if PLOT_DATA:
    plt.figure(0)
    plt.hist(means_re)
    plt.title("means real data")
    plt.grid()

    plt.figure(1)
    plt.hist(means_im)
    plt.title("means imag data")
    plt.grid()
    plt.show()


