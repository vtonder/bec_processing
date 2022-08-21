import matplotlib.pyplot as plt
import numpy as np
from constants import *

font = {'family': 'STIXGeneral',
        'size': 22}
plt.rc('font', **font)

# TODO: Finish up plot on time vs mean and std err of mean as error bar (this will be for only 1 channel)
# TODO: Create a plot of freq ch vs mean of means and std err of mean as error bar

frequencies = np.arange(856, 1712, freq_resolution)

class mean_stats():
    def __init__(self, directory, fn_means_re, fn_means_im, fn_std_err_re=None, fn_std_err_im=None,
                 fn_median_re=None, fn_median_im=None, fn_outlier_re=None, fn_outlier_im=None,
                 fn_bias_re=None, fn_bias_im=None):
        self.mean_re = np.load(directory + fn_means_re)
        self.mean_im = np.load(directory + fn_means_im)
        # mean of means
        self.means_mean_re, self.means_mean_im = [], []
        self.med_re, self.med_im = [], []
        self.max_re, self.max_im = [], []
        self.min_re, self.min_im = [], []
        self.p_75_re, self.p_75_im = [], []
        self.p_25_re, self.p_25_im = [], []
        self.var_mean_re, self.var_mean_im = [], []
        self.bias_re, self.bias_im = [], []

        if fn_std_err_re and fn_std_err_im:
            self.std_err_re = np.load(directory + fn_std_err_re)
            self.std_err_im = np.load(directory + fn_std_err_im)
            self.mean_std_err_re, self.mean_std_err_im = [], []

        if fn_median_re and fn_median_im:
            self.median_re = np.load(directory + fn_median_re)
            self.median_im = np.load(directory + fn_median_im)
            self.mean_median_re, self.mean_median_im = [], []

        # TODO: do they need to be a part of this class?
        if fn_outlier_re and fn_outlier_im:
            self.outlier_re = np.load(directory + fn_outlier_re)
            self.outlier_im = np.load(directory + fn_outlier_im)

        if fn_bias_re and fn_bias_im:
            self.bias_re = np.load(directory + fn_bias_re)
            self.bias_im = np.load(directory + fn_bias_im)

    def process_mean_stats(self):
        for ch in np.arange(1024):
            self.means_mean_re.append(np.mean(self.mean_re[ch,:]))
            self.med_re.append(np.median(self.mean_re[ch,:]))
            self.max_re.append(np.max(self.mean_re[ch,:]))
            self.min_re.append(np.min(self.mean_re[ch,:]))
            self.p_75_re.append(np.percentile(self.mean_re[ch,:], 75))
            self.p_25_re.append(np.percentile(self.mean_re[ch,:], 25))
            self.var_mean_re.append(np.var(self.mean_re[ch,:]))

            self.means_mean_im.append(np.mean(self.mean_im[ch,:]))
            self.med_im.append(np.median(self.mean_im[ch,:]))
            self.max_im.append(np.max(self.mean_im[ch,:]))
            self.min_im.append(np.min(self.mean_im[ch,:]))
            self.p_75_im.append(np.percentile(self.mean_im[ch,:], 75))
            self.p_25_im.append(np.percentile(self.mean_im[ch,:], 25))
            self.var_mean_im.append(np.var(self.mean_im[ch,:]))
    def process_std_err_stats(self):
        for ch in np.arange(1024):
            self.mean_std_err_im.append(np.mean(self.std_err_im[ch, :]))
            self.mean_std_err_re.append(np.mean(self.std_err_re[ch,:]))

# N1 = 1 000 000
#n1000000s239 = mean_stats('/home/vereese/git/phd_data/mean_analysis/1569/',
#                          "means_re_0x_1024ch_1000000_239.npy", "means_im_0x_1024ch_1000000_239.npy",
#                          "std_err_re_0x_1024ch_1000000_239.npy", "std_err_im_0x_1024ch_1000000_239.npy",
#                          "median_re_0x_1024ch_1000000_239.npy", "median_im_0x_1024ch_1000000_239.npy")
#n1000000s239.process_mean_stats()
#
## N2 = 100 000
#n100000s2391 = mean_stats('/home/vereese/git/phd_data/mean_analysis/1569/',
#                          "means_re_0x_1024ch_100000_2391.npy", "means_im_0x_1024ch_100000_2391.npy")
#n100000s2391.process_mean_stats()
#n100000s239 = mean_stats('/home/vereese/git/phd_data/mean_analysis/1569/',
#                          "means_re_0x_1024ch_100000_239.npy", "means_im_0x_1024ch_100000_239.npy")
#n100000s239.process_mean_stats()
#
## N3 = 10 000
#n10000s23913 = mean_stats('/home/vereese/git/phd_data/mean_analysis/1569/',
#                          "means_re_0x_1024ch_10000_23913.npy", "means_im_0x_1024ch_10000_23913.npy")
#n10000s23913.process_mean_stats()
#
#outlier_re = np.load("/home/vereese/git/phd_data/mean_analysis/1064/1064_real_outliers.npy")
#outlier_im = np.load("/home/vereese/git/phd_data/mean_analysis/1064/1064_imag_outliers.npy")

bias_re = np.load('/home/vereese/git/phd_data/mean_analysis/1569/means_bias_re_0x_1024ch_19.npy')
bias_im = np.load('/home/vereese/git/phd_data/mean_analysis/1569/means_bias_im_0x_1024ch_19.npy')
#mean_re = np.load("means_re.npy")
#mean_im = np.load("means_im.npy")

#plt.figure(0)
#plt.plot(frequencies[1:], n1000000s239.means_mean_re[1:], label='mean N1S1')
#plt.plot(frequencies[1:], n100000s2391.means_mean_re[1:], '--',label='mean N2S2')
#plt.plot(frequencies[1:], n10000s23913.means_mean_re[1:], 'o',label='mean N3S3')
#plt.plot(frequencies[1:], n1000000s239.p_75_re[1:], label='75% N1S1')
#plt.plot(frequencies[1:], n100000s2391.p_75_re[1:], '--',label='75% N2S2')
#plt.plot(frequencies[1:], n10000s23913.p_75_re[1:], 'o',label='75% N3S3')
#plt.plot(frequencies[1:], n1000000s239.p_25_re[1:], label='25% N1S1')
#plt.plot(frequencies[1:], n100000s2391.p_25_re[1:], '--',label='25% N2S2')
#plt.plot(frequencies[1:], n10000s23913.p_25_re[1:], 'o',label='25% N3S3')
#plt.plot(frequencies[1:], mean_std_err_re[1:], label='mean of std error of mean')
#plt.plot(frequencies[1:], n1000000s239.med_re[1:], label='median')
#plt.plot(frequencies[1:], max_re[1:], label='max')
#plt.plot(frequencies[1:], min_re[1:], label='min')
#plt.plot(frequencies[1:], n1000000s239.p_75_re[1:], label='75 %')
#plt.plot(frequencies[1:], n1000000s239.p_25_re[1:], label='25 %')
#plt.title("N1=1000000, N2=100000, N3=10000, S1=239, S2=2391, S3=23913")
#plt.xlabel("Frequency MHz")
#plt.legend()
#plt.grid()

# TODO Plot the channels with RFI in it
num_digits = int(np.round(np.log(239130880)))
plt.figure(1)
plt.plot(np.log(np.asarray([2, 23, 239, 2391, 23913, 239130, 2391308, 23913088, 239130880])), bias_re[860,10:],label='GPS L1')
plt.plot(np.log(np.asarray([2, 23, 239, 2391, 23913, 239130, 2391308, 23913088, 239130880])), bias_re[512,10:],label='middle')
plt.xlabel("log(number of data points)")
plt.ylabel("mean")
plt.legend()
plt.grid()


"""plt.figure(1)
plt.plot(frequencies[1:], n1000000s239.means_mean_re[1:], label='mean N1S1')
plt.plot(frequencies[1:], n100000s239.means_mean_re[1:], '--',label='mean N2S1')
#plt.plot(frequencies[1:], n10000s23913.means_mean_re[1:], 'o',label='mean N3S3')
plt.plot(frequencies[1:], n1000000s239.p_75_re[1:], label='75% N1S1')
plt.plot(frequencies[1:], n100000s2391.p_75_re[1:], '--',label='75% N2S1')
#plt.plot(frequencies[1:], n10000s23913.p_75_re[1:], 'o',label='75% N3S3')
plt.plot(frequencies[1:], n1000000s239.p_25_re[1:], label='25% N1S1')
plt.plot(frequencies[1:], n100000s2391.p_25_re[1:], '--',label='25% N2S1')
#plt.plot(frequencies[1:], n10000s23913.p_25_re[1:], 'o',label='25% N3S3')
#plt.plot(frequencies[1:], mean_std_err_re[1:], label='mean of std error of mean')
#plt.plot(frequencies[1:], n1000000s239.med_re[1:], label='median')
#plt.plot(frequencies[1:], max_re[1:], label='max')
#plt.plot(frequencies[1:], min_re[1:], label='min')
#plt.plot(frequencies[1:], n1000000s239.p_75_re[1:], label='75 %')
#plt.plot(frequencies[1:], n1000000s239.p_25_re[1:], label='25 %')
plt.title("N1=1000000, N2=100000, S1=239")
plt.legend()
plt.grid()

plt.figure(0)
plt.plot(np.arange(0,1024), outlier_re)
plt.title("outliers Re")
plt.grid()

plt.figure(1)
plt.plot(np.arange(0,1024), outlier_im)
plt.title("outliers Im")
plt.grid()
"""
plt.show()


"""plt.figure(1)
plt.plot(frequencies[1:], n1000000s239.means_mean_im[1:], label='mean')
#plt.plot(frequencies[1:], mean_std_err_re[1:], label='mean of std error of mean}')
plt.plot(frequencies[1:], n1000000s239.med_im[1:], label='median')
plt.plot(frequencies[1:], n1000000s239.max_im[1:], label='max')
plt.plot(frequencies[1:], n1000000s239.min_im[1:], label='min')
plt.plot(frequencies[1:], n1000000s239.p_75_im[1:], label='75 %')
plt.plot(frequencies[1:], n1000000s239.p_25_im[1:], label='25 %')
plt.title("stats of means Im")
plt.legend()
plt.grid()

plt.figure(2)
plt.plot(frequencies[1:], n1000000s239.var_mean_re[1:])
plt.title("var of means Re")
plt.grid()

plt.figure(3)
plt.plot(frequencies[1:], n1000000s239.var_mean_im[1:])
plt.title("var of means Im")
plt.grid()

plt.figure(4)
plt.plot(np.arange(0,1024), outlier_re)
plt.title("outliers Re")
plt.grid()

plt.figure(5)
plt.plot(np.arange(0,1024), outlier_im)
plt.title("outliers Im")
plt.grid()

plt.show()

# 3d histogram

fig = plt.figure(figsize=[30,30])
ax = fig.add_subplot(111, projection='3d')
nbins = 50
for z in np.arange(1,1024):
    hist, bins = np.histogram(mean_re[z], bins=nbins)
    xs = (bins[:-1] + bins[1:])/2
    ax.bar(xs, hist, zs=reversed_frequencies[z], zdir='y', color='b', ec='b', alpha=0.8)

ax.set_xlabel('Mean values')
ax.set_ylabel('Frequency [MHz]')
ax.set_zlabel('Number of occurences')
ax.set_title('Re component')
plt.savefig('/home/vereese/Documents/PhD/thesis/Figures/re_mean', bbox_inches='tight')

fig1 = plt.figure(figsize=[30,30])
ax1 = fig1.add_subplot(111, projection='3d')
for z in np.arange(1,1024):
    hist, bins = np.histogram(mean_im[z], bins=nbins)
    xs = (bins[:-1] + bins[1:])/2
    ax1.bar(xs, hist, zs=reversed_frequencies[z], zdir='y', color='b', ec='b', alpha=0.8)

ax1.set_xlabel('Mean values')
ax1.set_ylabel('Frequency [MHz]')
ax1.set_zlabel('Number of occurences')
ax1.set_title('Im component')
plt.savefig('/home/vereese/Documents/PhD/thesis/Figures/im_mean', bbox_inches='tight')

plt.show()

# For plotting 1 channel only

plt.figure(0)
plt.hist(mean_re[856,:])
plt.title("Histogram means real data for ch 856")
plt.grid()

plt.figure(1)
plt.hist(mean_im[856,:])
plt.title("Histogram means imag data for ch 856")
plt.grid()

plt.figure(2)
plt.plot(mean_re[856,:], label='mean for ch 856')
plt.plot(std_err_re[856,:], 'o', label='std err for ch 856')
plt.plot(-1 * std_err_re[856,:], 'o', label='std err for ch 856')
plt.title("means + std re data")
plt.legend()
plt.grid()

plt.figure(3)
plt.plot(means_im, label='mean')
plt.plot(std_err_im, 'o', label='std err')
plt.plot(-1 * std_err_im, 'o', label='std err')
plt.title("means + std imag data")
plt.legend()
plt.grid()
plt.show()"""