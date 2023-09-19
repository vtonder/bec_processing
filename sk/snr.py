import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from constants import time_resolution

def snr_toa_un(file_names, pulse_w, pulse_start):
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

# Setup fonts and sizes for publication, based on page dimensions in inches
# This is tuned for LaTeX beamer slides
textwidth =  9.6 #128.0 / 25.4 #
textheight = 7 # 96.0 / 25.4
plt.rc('font', size=11, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=11, labelsize=11)
plt.rc(('xtick', 'ytick'), labelsize=11)
plt.rc('legend', fontsize=11)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

fn_median = ["median_smoothed_I4.npy"]

fn_no_mit = ["intensity_2210.npy"]

fn_sk_1349 =  ["itegrated_sk_intensity_M512_2210.npy", "itegrated_sk_intensity_M1024_2210.npy",
              "itegrated_sk_intensity_M2048_2210.npy", "itegrated_sk_intensity_M4096_2210.npy",
              "itegrated_sk_intensity_M8192_2210.npy", "itegrated_sk_intensity_M16384_2210.npy"]

fn_sk_02699 = ["itegrated_sk_pfa_1_intensity_M512_2210.npy", "itegrated_sk_pfa_1_intensity_M1024_2210.npy",
              "itegrated_sk_pfa_1_intensity_M2048_2210.npy", "itegrated_sk_pfa_1_intensity_M4096_2210.npy",
              "itegrated_sk_pfa_1_intensity_M8192_2210.npy", "itegrated_sk_pfa_1_intensity_M16384_2210.npy"]



fn_5sigma = ["var_threshold_intensity_M512_2210.npy", "var_threshold_intensity_M1024_2210.npy",
             "var_threshold_intensity_M2048_2210.npy", "var_threshold_intensity_M4096_2210.npy",
             "var_threshold_intensity_M8192_2210.npy", "var_threshold_intensity_M16384_2210.npy"]

pulse_start1 = 3042
pulse_stop1 = 3153
pulse_w1 = pulse_stop1 - pulse_start1

pulse_start2 = 3042

profile, snr, toa_un = snr_toa_un(fn_no_mit, pulse_w1, pulse_start1)
profiles_sk_1349, snr_sk_1349, toa_un_sk_1349 = snr_toa_un(fn_sk_1349, 117, pulse_start2)
profiles_sk_02699, snr_sk_02699, toa_un_sk_02699 = snr_toa_un(fn_sk_02699, 118, pulse_start2)
profiles_5sigma, snr_5sigma, toa_un_5sigma = snr_toa_un(fn_5sigma, pulse_w1, pulse_start1)
profile_median, snr_median, toa_median = snr_toa_un(fn_median, pulse_w1, pulse_start1)

p1 = (profile["intensity_2210.npy"]-0.8*10**14)/max(profile["intensity_2210.npy"]-0.8*10**14)
p2 = (profiles_sk_1349["itegrated_sk_intensity_M16384_2210.npy"]-0.32*10**14)/max(profiles_sk_1349["itegrated_sk_intensity_M16384_2210.npy"]-0.32*10**14)
p3 = (profiles_sk_02699["itegrated_sk_pfa_1_intensity_M16384_2210.npy"]-0.31*10**14)/max(profiles_sk_02699["itegrated_sk_pfa_1_intensity_M16384_2210.npy"]-0.31*10**14)
p4 = (profiles_5sigma["var_threshold_intensity_M2048_2210.npy"] - 0.46*10**14)/max(profiles_5sigma["var_threshold_intensity_M2048_2210.npy"] - 0.46*10**14)
p5 = (profile_median["median_smoothed_I4.npy"]-0.29*10**14)/max(profile_median["median_smoothed_I4.npy"]-0.29*10**14)
phi = np.arange(0,1,1/len(profile["intensity_2210.npy"]))
plt.figure(0) #, figsize=[22,16])
plt.plot(phi, p3+1.9, label="PFA = 2%")
plt.plot(phi, p2+1.5, label="PFA = 0.27%")
plt.plot(phi, p5+0.9, label="median")
plt.plot(phi, p4+0.5, label=">= 5$\sigma$")
plt.plot(phi, p1+0.1, label="None")
#plt.plot(profile["intensity_2210.npy"], label="none")
#plt.plot(profiles_sk_1349["itegrated_sk_intensity_M16384_2210.npy"], label="pfa=0.27")
#plt.plot(profiles_sk_02699["itegrated_sk_pfa_1_intensity_M16384_2210.npy"], label="pfa=2")
#plt.plot(profile_median["median_smoothed_I4.npy"], label="median")
#plt.plot(profiles_5sigma["var_threshold_intensity_M2048_2210.npy"], label="5sigma")
plt.grid()
plt.legend()
plt.xlim([0,1])
plt.ylim([0,3])
plt.xlabel("pulsar phase")
plt.ylabel("normalized pulsar intensity profile")
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/test_profiles.png', bbox_inches='tight')

M = [512, 1024, 2048, 4096, 8192, 16384]

plt.figure(1) #, figsize=[22,16])
plt.plot(M, snr_sk_02699, '-o', label="SK, PFA=2%", linewidth=2)
plt.plot(M, snr_sk_1349, '-o', label="SK, PFA=0.27%", linewidth=2)
plt.plot(M, snr_5sigma, '-o', label=">= 5 $\sigma$", linewidth=2)
plt.plot(M, snr_median*np.ones(len(M)), '-o', label="median", linewidth=2)
plt.plot(M, snr*np.ones(len(M)), '-o', label="None", linewidth=2)
plt.legend()
plt.grid()
plt.xlim([512, 16384])
plt.ylim([0,8000])
plt.xlabel("M values")
plt.ylabel("SNR")
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/snr.eps', bbox_inches='tight')

plt.figure(2) #, figsize=[22,16])
plt.plot(M, toa_un*np.ones(len(M)), '-o', label="None", linewidth=2)
plt.plot(M, toa_median*np.ones(len(M)), '-o', label="median", linewidth=2)
plt.plot(M, toa_un_5sigma, '-o', label=">= 5 $\sigma$", linewidth=2)
plt.plot(M, toa_un_sk_1349, '-o', label="SK, PFA=0.27%", linewidth=2)
plt.plot(M, toa_un_sk_02699, '-o', label="SK, PFA=2%", linewidth=2)
plt.ylim([0,0.1])
plt.legend()
plt.grid()
plt.xlim([512, 16384])
plt.xlabel("M values")
plt.ylabel("TOA uncertainty [$\mu$s]")
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/toa.eps', bbox_inches='tight')
plt.show()

