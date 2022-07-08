import h5py
import numpy as np
from hos import Bispectrum
import time
import sys
sys.path.append('..')
from constants import *
from matplotlib import pyplot as plt

# TODO: Parameterise CAPITAL and N, freq ch values
# TODO: Finish up plot on time vs mean and std err of mean as error bar (this will be for only 1 channel)
# TODO: Create a plot of freq ch vs mean of means and std err of mean as error bar

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
PLOT_DATA = False 
SAVE_DATA = False 

"""for i, arg in enumerate(sys.argv, start = 1):
    if arg == '-h':
        print("Argument 1: Frequency channel, default set to 856")
        print("Argument 2: Number of samples in a set")

try:
  opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
except getopt.GetoptError:
  print 'test.py -i <inputfile> -o <outputfile>'
  sys.exit(2)
for opt, arg in opts:
  if opt == '-h':
     print 'test.py -i <inputfile> -o <outputfile>'
     sys.exit()
  elif opt in ("-i", "--ifile"):
     inputfile = arg
  elif opt in ("-o", "--ofile"):
     outputfile = arg
print 'Input file is "', inputfile
print 'Output file is "', outputfile"""

#vela_y = h5py.File('/home/vereese/pulsar_data/1604641064_wide_tied_array_channelised_voltage_0y.h5', 'r')
vela_y = h5py.File('/home/vereese/pulsar_data/1604641234_wide_tied_array_channelised_voltage_0x.h5', 'r')
data = vela_y['Data/bf_raw'][...]
data_re = data[:,13634560:,0]
data_im = data[:,13634560:,1]
data_len = len(data_re[0,:]) # Re & Im freq channels will have the same length
num_ch = 1024 # number of frequency channels
N = 10000 # number of samples in a set
S = int(data_len/N) # number of sets
means_re, means_im = np.zeros([num_ch, S]), np.zeros([num_ch, S])
std_err_re, std_err_im = np.zeros([num_ch, S]), np.zeros([num_ch, S])

print("number of sets:", S)
print("shape re", np.shape(data_re))
print("shape im", np.shape(data_im))

for k in np.arange(1024):
    for i in np.arange(S):
        print("ch:",k," set num:",i)
        print("start:", i*N, " end:",(i+1)*N)
        means_re[k,i] = np.mean(data_re[k, i*N:(i+1)*N])
        means_im[k,i] = np.mean(data_im[k, i*N:(i+1)*N])
        std_err_re[k,i] = np.sqrt(np.var(data_re[k, i*N:(i+1)*N])/N)
        std_err_im[k,i] = np.sqrt(np.var(data_im[k, i*N:(i+1)*N])/N)

np.save("means_re", means_re)
np.save("means_im", means_im)
np.save("std_err_re", std_err_re)
np.save("std_err_im", std_err_im)

# These were only for 1 channel

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

print("std error of means re", np.sqrt(np.var(data_re[0:N])/N))
print("std error of means im", np.sqrt(np.var(data_im[0:N])/N))
print("std error of means re", np.sqrt(np.var(data_re[N:2*N])/N))
print("std error of means im", np.sqrt(np.var(data_im[N:2*N])/N))
print("")"""

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
    plt.title("Histogram means real data")
    plt.grid()

    plt.figure(1)
    plt.hist(means_im)
    plt.title("Histogram means imag data")
    plt.grid()

    plt.figure(2)
    plt.plot(means_re, label='mean')
    plt.plot(std_err_re,'o', label='std err')
    plt.plot(-1*std_err_re,'o', label='std err')
    plt.title("means + std re data")
    plt.legend()
    plt.grid()

    plt.figure(3)
    plt.plot(means_im, label='mean')
    plt.plot(std_err_im, 'o',label='std err')
    plt.plot(-1*std_err_im, 'o',label='std err')
    plt.title("means + std imag data")
    plt.legend()
    plt.grid()
    plt.show()

