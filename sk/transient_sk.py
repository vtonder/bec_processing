import numpy as np
import math
from pulsar_snr import PI, get_profile, get_pho, compute 
from kurtosis import sk_gaus_tran
import sys
sys.path.append('../')
from constants import num_ch, lower_limit_1s, frequencies, thesis_font, a4_textwidth, a4_textheight
import argparse
from matplotlib import pyplot as plt

# Setup fonts and sizes for publication, based on page dimensions in inches
textwidth =  a4_textwidth
textheight = a4_textheight
font_size = thesis_font
# groups are like plt.figure plt.legend etc
plt.rc('font', size=font_size, family='serif')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=font_size, labelsize=font_size)
plt.rc(('xtick', 'ytick'), labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('lines', markersize=5)
# The following should only be used for beamer
# plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
figheight = 0.65 * textwidth
plt.rc('mathtext', fontset='cm')
# to get this working needed to do: sudo apt install cm-super
plt.rc("text", usetex = True)
plt.rc("figure", figsize = (textwidth, figheight))

parser = argparse.ArgumentParser()
parser.add_argument("-d", dest = "dir", help = "directory where data is located. default location: /home/vereese/git/phd_data/sk_analysis/2210/", default = "/home/vereese/git/phd_data/sk_analysis/2210/")
# "/home/vereese/git/phd_data/sk_analysis/2210/4sig/"
args = parser.parse_args()
DIR = args.dir

DEBUG = False

intensity = PI(DIR, "intensity_z_2210_p45216.npy", "num_nz_z_2210_p45216.npy", initialise=False)
intensity.compute()

#mean_shift_ito_1sigma = {8192:[], 4096:[], 2048:[], 1024:[], 512:[], 256:[], 128:[], 64:[]}
mean_shift_ito_1sigma = {4096:[], 2048:[], 1024:[], 512:[], 256:[]}
fig, ax = plt.subplots()
for M, ms in mean_shift_ito_1sigma.items():
    P_M = intensity.samples_T / M # pulsar period / M This is needed to adjust predicted M because not all M windows will contain the on-pulse section
    # note for loop skips over masked channels
    #for i in np.arange(51, num_ch-50):
    for i in np.arange(num_ch):
        #if 95 <= i <= 126:
        #    continue
        I_ch = intensity.I[i, :].reshape(1, intensity.samples_T) # channel i's intensity
        nz_ch = intensity.nz[i, :].reshape(1, intensity.samples_T) # number of non zero data points that went into summation for channel i
        profile = get_profile(I_ch, nz_ch)
        pulse_start, pulse_stop, pulse_width, snr, toa_un = compute(I_ch, nz_ch)
        delta = pulse_width / M
        bop = math.ceil(delta) # blocks containing on-pulse phase of pulsar. know for M = 512 there is only 1 M block that contains the on pulse out of the ~9.4 M = 512 blocks that's in the pulse period
        pho = get_pho(profile)
        sk_offset = sk_gaus_tran(pho, delta)
        predicted_sk = ((bop * sk_offset) + ((P_M - bop) * 1)) / P_M
        s1 = 1 - lower_limit_1s[M]
        ms.append((np.abs(1-predicted_sk) / s1) * 100)
        if DEBUG:
            print("Pulse width: ", pulse_width)
            print("Pulse period / M: ", P_M)
            print("Number of M blocks that contain on pulse: ", bop)
            print("SK in block that contains on pulse: ", sk_offset)
            print("predicted sk: ", predicted_sk)
            print("lower threshold 1 sigma away from mean 1: ", s1)
            print("mean sk shift ito 1sigma : ", (np.abs(1 - predicted_sk) / s1) * 100)

    ax.plot(frequencies, ms, label="$M$ = " + str(M), linewidth=2)


ax.set_ylabel("\% shift in $\overline{SK}$ ito 1$\sigma$ threshold")
ax.set_xlabel("frequency [MHz]")
ax.set_xlim([frequencies[0], frequencies[-1]])
plt.legend()
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/transient_sk.pdf', bbox_inches='tight')
plt.show()