import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from constants import frequencies, thesis_font, a4_textwidth, a4_textheight, lower_limit_3s, upper_limit_3s
import argparse

# A script to plot the mean SK across the observation for J0437-4715

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

args = parser.parse_args()
DIR = args.dir

#skx_M64 = np.load(DIR + "sk_z_M64_m1_n1_2210_0x.npy")
#sky_M64 = np.load(DIR + "sk_z_M64_m1_n1_2210_0y.npy")
skx_M256 = np.load(DIR + "sk_z_M256_m1_n1_2210_0x.npy")
sky_M256 = np.load(DIR + "sk_z_M256_m1_n1_2210_0y.npy")
skx_M2048 = np.load(DIR + "sk_z_M2048_m1_n1_2210_0x.npy")
sky_M2048 = np.load(DIR + "sk_z_M2048_m1_n1_2210_0y.npy")

#m_skx_M64 = skx_M64.mean(axis = 1)
#m_sky_M64 = sky_M64.mean(axis = 1)
m_skx_M256 = skx_M256.mean(axis = 1)
m_sky_M256 = sky_M256.mean(axis = 1)
m_skx_M2048 = skx_M2048.mean(axis = 1)
m_sky_M2048 = sky_M2048.mean(axis = 1)

fig, ax = plt.subplots()
ax.plot(frequencies, m_sky_M2048, label="V-pol, $M$ = 2048", linewidth=2)
ax.plot(frequencies, m_skx_M2048, label="H-pol, $M$ = 2048", linewidth=2)
ax.plot(frequencies, m_sky_M256, label="V-pol, $M$ = 256", linewidth=2)
ax.plot(frequencies, m_skx_M256, label="H-pol, $M$ = 256", linewidth=2)
#ax.semilogy(frequencies, m_sky_M2048, label="V-pol, $M$ = 2048", linewidth=2)
#ax.semilogy(frequencies, m_skx_M2048, label="H-pol, $M$ = 2048", linewidth=2)
#ax.semilogy(frequencies, m_sky_M256, label="V-pol, $M$ = 256", linewidth=2)
#ax.semilogy(frequencies, m_skx_M256, label="H-pol, $M$ = 256", linewidth=2)
#ax.hlines(lower_limit_3s[256], frequencies[0], frequencies[-1], linestyle = "--", color="red", label="$\pm3\sigma$ thresholds, M=256")
#ax.hlines(upper_limit_3s[256], frequencies[0], frequencies[-1], linestyle = "--", color="red")
#ax.hlines(lower_limit_3s[2048], frequencies[0], frequencies[-1], linestyle = "--", color="blue", label="$\pm3\sigma$ thresholds, M=2048")
#ax.hlines(upper_limit_3s[2048], frequencies[0], frequencies[-1], linestyle = "--", color="blue")
ax.hlines(1, frequencies[0], frequencies[-1], linestyle='--', color="black", label="$\overline{SK} = 1$") #, linewidth=2)
#ax.plot(frequencies, m_sky_M64, label="V-pol, M = 64")
#ax.plot(frequencies, m_skx_M64, label="H-pol, M = 64")
ax.set_xlabel("frequency [MHz]")
ax.set_ylabel("$\overline{SK}$")
ax.set_xlim([frequencies[0], frequencies[-1]])
ax.set_ylim([0.995, 1.015])
#ax.set_ylim([0.6, 1.55])
plt.legend()
plt.savefig('/home/vereese/thesis_pics/mean_sk1.pdf', transparent=True, bbox_inches='tight')
plt.show()

