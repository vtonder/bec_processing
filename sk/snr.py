import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from constants import time_resolution

def snr_toa_un(file_names):
    DIR = "/home/vereese/git/phd_data/sk_analysis/2210/"

    power_spectrum = {}
    profile = {}
    snrs = []
    toa_uns = []
    for fn in file_names:
        ps = np.load(DIR + fn)
        p = ps.sum(axis=0)
        m = np.mean(p[0:pulse_w])
        s = np.std(p[0:pulse_w])
        power_spectrum.update({fn: ps})
        profile.update({fn: p})
        snr = 0
        for i in np.arange(pulse_w):
            snr = snr + (p[pulse_start + i] - m)
        snr = snr / (s * np.sqrt(pulse_w))
        snrs.append(snr)
        toa_uns.append((pulse_w * time_resolution) / snr)
        print("Analysis for: ", fn)
        print("SNR         : ", snr)
        print("TOA un [us] : ", (pulse_w * time_resolution) / snr)

    return profile, snrs, toa_uns


#font = {'family': 'STIXGeneral',
#        'size': 42}
#plt.rc('font', **font)

# Setup fonts and sizes for publication, based on page dimensions in inches
# This is tuned for LaTeX beamer slides
textwidth = 9.6 #128.0 / 25.4
textheight = 7 #96.0 / 25.4
plt.rc('font', size=22, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=22, labelsize=22)
plt.rc(('xtick', 'ytick'), labelsize=22)
plt.rc('legend', fontsize=22)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

#plt.rc('mathtext', fontset='custom',
#       rm='Source Sans Pro', bf='Source Sans Pro:bold',
#       it='Source Sans Pro:italic', tt='Source Code Pro')

fn_median = ["median_smoothed_I.npy"]

fn_no_mit = ["intensity_2210.npy"]

fn_sk_1349 =  ["itegrated_sk_intensity_M512_2210.npy", "itegrated_sk_intensity_M1024_2210.npy",
              "itegrated_sk_intensity_M2048_2210.npy", "itegrated_sk_intensity_M4096_2210.npy",
              "itegrated_sk_intensity_M8192_2210.npy", "itegrated_sk_intensity_M16384_2210.npy"]

fn_sk_02699 = ["itegrated_sk_pfa_02699_intensity_M512_2210.npy", "itegrated_sk_pfa_02699_intensity_M1024_2210.npy",
              "itegrated_sk_pfa_02699_intensity_M2048_2210.npy", "itegrated_sk_pfa_02699_intensity_M4096_2210.npy",
              "itegrated_sk_pfa_02699_intensity_M8192_2210.npy", "itegrated_sk_pfa_02699_intensity_M16384_2210.npy"]

fn_5sigma = ["var_threshold_intensity_M512_2210.npy", "var_threshold_intensity_M1024_2210.npy",
             "var_threshold_intensity_M2048_2210.npy", "var_threshold_intensity_M4096_2210.npy",
             "var_threshold_intensity_M8192_2210.npy", "var_threshold_intensity_M16384_2210.npy"]

pulse_start = 2900
pulse_stop = 3250
pulse_w = pulse_stop - pulse_start

profile, snr, toa_un = snr_toa_un(fn_no_mit)
profiles_sk_1349, snr_sk_1349, toa_un_sk_1349 = snr_toa_un(fn_sk_1349)
profiles_sk_02699, snr_sk_02699, toa_un_sk_02699 = snr_toa_un(fn_sk_02699)
profiles_5sigma, snr_5sigma, toa_un_5sigma = snr_toa_un(fn_5sigma)
profile_median, snr_median, toa_median = snr_toa_un(fn_median)

mp = list(profile["intensity_2210.npy"]).index(max(list(profile["intensity_2210.npy"])))
print("max point index: ", mp)
num_2_roll = int(len(profile["intensity_2210.npy"])/2 - mp)
print("roll by", num_2_roll)
labels = ["None", "SK, M = 2048, PFA=0.1349%", ]
phi = np.arange(0,1,1/len(profile["intensity_2210.npy"]))
plt.figure(0) #, figsize=[22,16])
plt.plot(phi, np.roll(profile["intensity_2210.npy"], num_2_roll), label="None")
plt.plot(phi, np.roll(profiles_sk_1349["itegrated_sk_intensity_M2048_2210.npy"], num_2_roll), label="PFA = 0.1349%")
plt.plot(phi, np.roll(profiles_sk_02699["itegrated_sk_pfa_02699_intensity_M2048_2210.npy"], num_2_roll), label="PFA = 0.2699%")
plt.plot(phi, np.roll(profiles_5sigma["var_threshold_intensity_M2048_2210.npy"], num_2_roll), label=">= 5$\sigma$")
plt.plot(phi, np.roll(profile_median["median_smoothed_I.npy"], num_2_roll), label="median")
plt.grid()
plt.legend(loc='right')
plt.xlim([0,1])
plt.xlabel("pulsar phase")
plt.ylabel("pulsar profile")
plt.savefig('/home/vereese/Documents/PhD/jai-2e/profiles2.eps', bbox_inches='tight')

M = [512, 1024, 2048, 4096, 8192, 16384]
plt.figure(1) #, figsize=[22,16])
plt.plot(M, snr*np.ones(len(M)), label="None", linewidth=2)
plt.plot(M, snr_sk_1349, label="SK, PFA=0.1349%", linewidth=2)
plt.plot(M, snr_sk_02699, label="SK, PFA=0.2699%", linewidth=2)
plt.plot(M, snr_5sigma, label=">= 5 $\sigma$", linewidth=2)
plt.plot(M, snr_median*np.ones(len(M)), label="median", linewidth=2)
plt.legend(loc='right')
plt.grid()
plt.xlim([512, 16384])
plt.xlabel("M values")
plt.ylabel("SNR")
plt.savefig('/home/vereese/Documents/PhD/jai-2e/snr.eps', bbox_inches='tight')

plt.figure(2) #, figsize=[22,16])
plt.plot(M, toa_un*np.ones(len(M)), label="None", linewidth=2)
plt.plot(M, toa_un_sk_1349, label="SK, PFA=0.1349%", linewidth=2)
plt.plot(M, toa_un_sk_02699, label="SK, PFA=0.2699%", linewidth=2)
plt.plot(M, toa_un_5sigma, label=">= 5 $\sigma$", linewidth=2)
plt.plot(M, toa_median*np.ones(len(M)), label="median", linewidth=2)
plt.legend(loc='right')
plt.grid()
plt.xlim([512, 16384])
plt.xlabel("M values")
plt.ylabel("TOA uncertainty")
plt.savefig('/home/vereese/Documents/PhD/jai-2e/toa.eps', bbox_inches='tight')
plt.show()

