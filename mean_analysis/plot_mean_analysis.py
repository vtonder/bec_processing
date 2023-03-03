import matplotlib.pyplot as plt
import numpy as np
from constants import gps_l1_ch, gps_l2_ch, gal_e6_ch, gal_5b_ch, h1_ch, frequencies

font = {'family': 'STIXGeneral',
        'size': 26}
plt.rc('font', **font)

# TODO: Finish up plot on time vs mean and std err of mean as error bar (this will be for only 1 channel)
# TODO: Create a plot of freq ch vs mean of means and std err of mean as error bar

class mean_stats():
    def __init__(self, directory, fn_means_re=None, fn_means_im=None, fn_std_err_re=None, fn_std_err_im=None,
                 fn_median=None, fn_outlier_re=None, fn_outlier_im=None,
                 fn_bias_re=None, fn_bias_im=None):

        if fn_means_re and fn_means_im:
            self.mean_re = np.load(directory + fn_means_re)
            self.mean_im = np.load(directory + fn_means_im)

        if fn_std_err_re and fn_std_err_im:
            self.std_err_re = np.load(directory + fn_std_err_re)
            self.std_err_im = np.load(directory + fn_std_err_im)

        if fn_median:
            self.medians = np.load(directory + fn_median)

        # TODO: do they need to be a part of this class?
        if fn_outlier_re and fn_outlier_im:
            self.outlier_re = np.load(directory + fn_outlier_re)
            self.outlier_im = np.load(directory + fn_outlier_im)

        if fn_bias_re and fn_bias_im:
            self.bias_re = np.load(directory + fn_bias_re)
            self.bias_im = np.load(directory + fn_bias_im)

    def process_mean_stats(self):
        self.mean_means_re = np.mean(self.mean_re, axis=1)
        self.med_means_re = np.median(self.mean_re, axis=1)
        self.max_means_re = np.max(self.mean_re, axis=1)
        self.min_means_re = np.min(self.mean_re, axis=1)
        self.p_75_means_re = np.percentile(self.mean_re, 75, axis=1)
        self.p_25_means_re = np.percentile(self.mean_re, 25, axis=1)
        self.var_means_re = np.var(self.mean_re, axis=1)

        self.mean_means_im = np.mean(self.mean_im, axis=1)
        self.max_means_im = np.max(self.mean_im, axis=1)
        self.min_means_im = np.min(self.mean_im, axis=1)
        self.p_75_means_im = np.percentile(self.mean_im, 75, axis=1)
        self.p_25_means_im = np.percentile(self.mean_im, 25, axis=1)
        self.var_means_im = np.var(self.mean_im, axis=1)

    def process_median(self):
        self.mean_med_re = np.mean(self.medians[:,:,0], axis=1)
        self.med_med_re = np.median(self.medians[:,:,0], axis=1)
        self.p_75_med_re = np.percentile(self.medians[:, :, 0], 75, axis=1)
        self.p_25_med_re = np.percentile(self.medians[:, :, 0], 25, axis=1)


    def process_std_err_stats(self):
        self.mean_std_err_im = np.mean(self.std_err_im, axis=1)
        self.mean_std_err_re = np.mean(self.std_err_re, axis=1)

n1000000s239 = mean_stats('/home/vereese/git/phd_data/mean_analysis/1569/',
                          "means_re_0x_1024ch_1000000_239.npy", "means_im_0x_1024ch_1000000_239.npy",
                          "std_err_re_0x_1024ch_1000000_239.npy", "std_err_im_0x_1024ch_1000000_239.npy",
                          "medians_0x_1024ch_1000000_239.npy")
n1000000s239.process_mean_stats()
n1000000s239.process_std_err_stats()
n1000000s239.process_median()

n100000s2391 = mean_stats('/home/vereese/git/phd_data/mean_analysis/1569/',
                          "means_re_0x_1024ch_100000_2391.npy", "means_im_0x_1024ch_100000_2391.npy")
n100000s2391.process_mean_stats()
n100000s239 = mean_stats('/home/vereese/git/phd_data/mean_analysis/1569/',
                          "means_re_0x_1024ch_100000_239.npy", "means_im_0x_1024ch_100000_239.npy")
n100000s239.process_mean_stats()

n10000s23913 = mean_stats('/home/vereese/git/phd_data/mean_analysis/1569/',
                          "means_re_0x_1024ch_10000_23913.npy", "means_im_0x_1024ch_10000_23913.npy")
n10000s23913.process_mean_stats()
n10000s239 = mean_stats('/home/vereese/git/phd_data/mean_analysis/1569/',
                        "means_re_0x_1024ch_10000_239.npy", "means_im_0x_1024ch_10000_239.npy")
n10000s239.process_mean_stats()

outlier_re = np.load("/home/vereese/git/phd_data/mean_analysis/1064/1064_real_outliers.npy")
outlier_im = np.load("/home/vereese/git/phd_data/mean_analysis/1064/1064_imag_outliers.npy")

bias_re = np.load('/home/vereese/git/phd_data/mean_analysis/1569/means_bias_re_0x_1024ch_9.npy')
bias_im = np.load('/home/vereese/git/phd_data/mean_analysis/1569/means_bias_im_0x_1024ch_9.npy')
"""
# Mean of means become less because use more data points
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=[15,12])
ax1.plot(frequencies[1:], n10000s239.mean_means_re[1:],   label='N1=2.86 s')
ax1.plot(frequencies[1:], n100000s239.mean_means_re[1:],  label='N2=28.6 s')
ax1.plot(frequencies[1:], n1000000s239.mean_means_re[1:], label='N3=286 s')
ax2.plot(frequencies[1:], n10000s239.mean_means_im[1:],   label='N1=2.86 s')
ax2.plot(frequencies[1:], n100000s239.mean_means_im[1:],  label='N2=28.6 s')
ax2.plot(frequencies[1:], n1000000s239.mean_means_im[1:], label='N3=286 s')
ax1.set_xlim([frequencies[1], frequencies[-1]])
ax2.set_xlim([frequencies[1], frequencies[-1]])
ax1.grid()
ax1.legend()
ax2.grid()
ax2.legend()
ax2.set_xlabel("Frequency MHz")
ax1.set_ylabel("Real Mean")
ax2.set_ylabel("Imag Mean")
plt.savefig('/home/vereese/Documents/PhD/CASPER2022/presentation/means', bbox_inches='tight')
plt.show()
"""

#Mean of means stay constant because use the same number of data points and therefore mean would stay the same
plt.figure(0)
plt.plot(frequencies[1:], n10000s23913.mean_means_re[1:], label='N=  10 000, S=23913')
plt.plot(frequencies[1:], n100000s2391.mean_means_re[1:], label='N= 100 000, S=2391')
plt.plot(frequencies[1:], n1000000s239.mean_means_re[1:], label='N=1000 000, S=239')
plt.title("Mean of means. Same total length")
plt.legend()
plt.xlabel("Frequency MHz")
plt.grid()

plt.figure(1,figsize=[15,12])
#for i in range(1,1024):
#    plt.plot([10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000], bias_re[i])#,label='GAL 5B')
plt.plot([10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000], bias_re[gal_5b_ch],label='GAL 5B')
plt.plot([10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000], bias_re[590],label='1350 MHz')
plt.plot([10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000], bias_re[h1_ch],label='HI')
plt.plot([10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000], bias_re[gps_l1_ch],label='GPS L1')
plt.xlim([10, 1000000000])
plt.xscale('log',base=10)
plt.yscale('log',base=10)
plt.xlabel("Number of samples")
plt.ylabel("Mean")
plt.legend()
plt.grid(which="both")
#plt.savefig('/home/vereese/Documents/PhD/CASPER2022/presentation/dcBias_allch', bbox_inches='tight')

plt.figure(2, figsize=[15,12])
plt.plot(frequencies[1:], n10000s239.p_75_means_re[1:], 'b', label='N1 S1')
plt.plot(frequencies[1:], n10000s239.p_25_means_re[1:], 'b')
plt.plot(frequencies[1:], n10000s23913.p_75_means_re[1:], 'c', label='N1 S3')
plt.plot(frequencies[1:], n10000s23913.p_25_means_re[1:], 'c')

plt.plot(frequencies[1:], n100000s239.p_75_means_re[1:], 'g', label='N2 S1')
plt.plot(frequencies[1:], n100000s239.p_25_means_re[1:], 'g')
plt.plot(frequencies[1:], n100000s2391.p_75_means_re[1:], 'y', label='N2 S2')
plt.plot(frequencies[1:], n100000s2391.p_25_means_re[1:], 'y')

plt.plot(frequencies[1:], n1000000s239.p_75_means_re[1:], 'r', label='N3 S1')
plt.plot(frequencies[1:], n1000000s239.p_25_means_re[1:], 'r')

plt.xlim([frequencies[1], frequencies[-1]])
plt.title("N1=12ms, N2=0.12s, N3=1.2s, S1=239, S2=2391, S3=23913")
plt.legend()
plt.ylabel("25th and 75th % of means")
plt.xlabel("Frequency MHz")
plt.grid()
#plt.savefig('/home/vereese/Documents/PhD/CASPER2022/presentation/percentiles', bbox_inches='tight')

plt.figure(3)
plt.plot(frequencies[1:], n10000s239.p_75_means_im[1:], 'b', label='N1 S1')
plt.plot(frequencies[1:], n10000s239.p_25_means_im[1:], 'b')
plt.plot(frequencies[1:], n10000s23913.p_75_means_im[1:], 'c', label='N1 S3')
plt.plot(frequencies[1:], n10000s23913.p_25_means_im[1:], 'c')

plt.plot(frequencies[1:], n100000s239.p_75_means_im[1:], 'g', label='N2 S1')
plt.plot(frequencies[1:], n100000s239.p_25_means_im[1:], 'g')
plt.plot(frequencies[1:], n100000s2391.p_75_means_im[1:], 'y', label='N2 S2')
plt.plot(frequencies[1:], n100000s2391.p_25_means_im[1:], 'y')

plt.plot(frequencies[1:], n1000000s239.p_75_means_im[1:], 'r', label='N3 S1')
plt.plot(frequencies[1:], n1000000s239.p_25_means_im[1:], 'r')

plt.xlim([frequencies[1], frequencies[-1]])
plt.title("N1=12ms, N2=0.12s, N3=1.2s, S1=239, S2=2391, S3=23913")
plt.legend()
plt.ylabel("25th and 75th % of means")
plt.xlabel("Frequency MHz")
plt.grid()

plt.figure(4)
plt.plot(frequencies[1:], n1000000s239.mean_means_re[1:]/n1000000s239.mean_std_err_re[1:] , label='mean of means/ std error of means')
plt.plot(frequencies[1:], n1000000s239.mean_means_re[1:]/np.sqrt(n1000000s239.var_means_re[1:]) , label='mean of means/ std dev of means')
plt.title("mean to std error relation")
plt.legend()
plt.xlabel("Frequency MHz")
plt.grid()

plt.figure(5)
plt.plot(frequencies[1:], n1000000s239.med_means_re[1:], label='median(means)')
plt.plot(frequencies[1:], n1000000s239.med_med_re[1:], label='median(medians)')
plt.plot(frequencies[1:], n1000000s239.mean_med_re[1:], label='mean(medians)')
plt.plot(frequencies[1:], n1000000s239.p_25_med_re[1:], label='25th(medians)')
plt.plot(frequencies[1:], n1000000s239.p_75_med_re[1:], label='75th(medians)')
plt.plot(frequencies[1:], n1000000s239.var_means_re[1:], label='var(means)')
plt.xlabel("Frequency MHz")
plt.legend()
plt.title("median")
plt.grid()

plt.figure(6)
plt.plot(frequencies[1:], n1000000s239.med_means_re[1:], label='median(means)')
plt.plot(frequencies[1:], n1000000s239.mean_means_re[1:], label='mean(means)')
plt.xlabel("Frequency MHz")
plt.legend()
plt.title("means")
plt.grid()

# Outliers are 0 around 1200MHz becausestd is so high because the data hits the rails all the time
plt.figure(7)
plt.plot(frequencies[1:], outlier_re[1:])
plt.title("outliers Re")
plt.grid()

plt.figure(8)
plt.plot(frequencies[1:], outlier_im[1:])
plt.title("outliers Im")
plt.grid()

plt.figure(9)
plt.plot(frequencies[1:], n1000000s239.var_means_re[1:])
plt.title("var of means Re")
plt.grid()

plt.figure(10)
plt.plot(frequencies[1:], n1000000s239.var_means_im[1:])
plt.title("var of means Im")
plt.grid()

plt.show()

"""
# 3d histogram
# WARNING: need to modify for this part to run again. ie 
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

"""