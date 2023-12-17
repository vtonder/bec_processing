import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from constants import thesis_font, a4_textwidth, a4_textheight

# Plots results from gain.py and from hacking intensity.py to fold 2210 data at GSM frequencies

# references:
# https://www.rfcafe.com/references/electrical/gsm-specs.htm

# Setup fonts and sizes for publication, based on page dimensions in inches
textwidth =  a4_textwidth
textheight = a4_textheight
font_size = thesis_font
plt.rc('font', size=font_size, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=font_size, labelsize=font_size)
plt.rc(('xtick', 'ytick'), labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

gain1 = np.load("/home/vereese/git/phd_data/sk_analysis/gain/gain_0.1_rfi_l4sigu4sig_M1024_d4096000000.npy")
gain2 = np.load("/home/vereese/git/phd_data/sk_analysis/gain/gain_0.1_rfi_l4sigu4sig_M2048_d4096000000.npy")
gain3 = np.load("/home/vereese/git/phd_data/sk_analysis/gain/gain_0.1_rfi_l4sigu4sig_M4096_d4096000000.npy")

gsmx216 = np.load("/home/vereese/git/phd_data/sk_analysis/gain/gsm_216.667_intensity_xpol_2210.npy")
gsmy216 = np.load("/home/vereese/git/phd_data/sk_analysis/gain/gsm_216.667_intensity_ypol_2210.npy")

gsmx1733 = np.load("/home/vereese/git/phd_data/sk_analysis/gain/gsm_1733.333_intensity_xpol_2210.npy")
gsmy1733 = np.load("/home/vereese/git/phd_data/sk_analysis/gain/gsm_1733.333_intensity_ypol_2210.npy")

ts_freq = 1733.33     # GSM time slot frequency
ts_T = 1.0 / ts_freq
ts_t = np.arange(0, ts_T, ts_T / gsmx1733.shape[1])

df_freq = 216.667     # GSM data frame frequency
df_T = 1.0 / df_freq
df_t = np.arange(0, df_T, df_T / gsmx216.shape[1])

normx216, normy216 = np.zeros(gsmx216.shape), np.zeros(gsmy216.shape)
normx1733, normy1733 = np.zeros(gsmx1733.shape), np.zeros(gsmy1733.shape)

medx216 = np.median(gsmx216, axis=1)
medy216 = np.median(gsmy216, axis=1)
medx1733 = np.median(gsmx1733, axis=1)
medy1733 = np.median(gsmy1733, axis=1)

for i in np.arange(1024):
    normx216[i, :] = gsmx216[i, :] / medx216[i]
    normy216[i, :] = gsmy216[i, :] / medy216[i]
    normx1733[i, :] = gsmx1733[i, :] / medx1733[i]
    normy1733[i, :] = gsmy1733[i, :] / medy1733[i]

fig, ax = plt.subplots()
ax.semilogy(gain1[0,:], gain1[1,:], 'o', label="M = 1024")
ax.semilogy(gain2[0,:], gain2[1,:], 'o', label="M = 2048")
ax.semilogy(gain3[0,:], gain3[1,:], 'o', label="M = 4096")
ax.set_ylabel("% RFI flagged")
ax.set_xlabel("% Power increase")
ax.set_title("PFA = 4$\sigma$, RFI at 10% duty cycle")
#ax.set_ylim((0,0.5))
#ax.set_xlim((0,6.5))
plt.grid()
plt.legend()
#plt.savefig('/home/vereese/Documents/PhD//gain.eps', bbox_inches='tight')

fig1, ax1 = plt.subplots()
ax1.imshow(normx216, origin="lower", aspect="auto", vmin=0.99, vmax=1.01)
ax1.set_title("H-pol, fold freq = 216 Hz")

fig2, ax2 = plt.subplots()
ax2.imshow(normy216, origin="lower", aspect="auto", vmin=0.99, vmax=1.01)
ax2.set_title("V-pol, fold freq = 216 Hz")

fig3, ax3 = plt.subplots()
ax3.plot(df_t*1000, normy216[550:750,:].sum(axis=0))
ax3.set_xlabel("time ms")
ax3.set_ylabel("power variation V pol")
ax3.set_title("V-pol, fold freq = 216 Hz, ch 550-750 (clean band)")

fig33, ax33 = plt.subplots()
ax33.plot(df_t*1000, normx216[550:750,:].sum(axis=0))
ax33.set_xlabel("time ms")
ax33.set_ylabel("power variation H pol")
ax33.set_title("H-pol, fold freq = 216 Hz, ch 550-750 (clean band)")

fig4, axs4 = plt.subplots()
axs4.imshow(normx1733, origin="lower", aspect="auto", vmin=0.99, vmax=1.01, extent=[0, ts_T*1000, 856, 1712])
axs4.set_xlabel("GSM timeslot phase [ms]")
axs4.set_ylabel("frequency [MHz]")
#plt.savefig('/home/vereese/Documents/PhD/CASPER2023/casper_presentation/gsm1773hpol.eps', bbox_inches='tight')

fig5, axs5 = plt.subplots()
axs5.imshow(normy1733, origin="lower", aspect="auto", vmin=0.99, vmax=1.01, extent=[0, ts_T*1000, 856, 1712])
axs5.set_xlabel("GSM timeslot phase [ms]")
axs5.set_ylabel("frequency [MHz]")
#plt.savefig('/home/vereese/Documents/PhD/CASPER2023/casper_presentation/gsm1773vpol.eps', bbox_inches='tight')

fig6, ax6 = plt.subplots()
#ax6.plot(ts_t*1000, normy1733[550:750,:].sum(axis=0), label="V-pol", linewidth=2)
#ax6.plot(ts_t*1000, normx1733[550:750,:].sum(axis=0), label="H-pol", linewidth=2)
ax6.plot(ts_t*1000, normy1733.sum(axis=0), label="V-pol", linewidth=2)
ax6.plot(ts_t*1000, normx1733.sum(axis=0), label="H-pol", linewidth=2)
ax6.set_xlim([ts_t[0]*1000, ts_t[-1]*1000])
ax6.set_xlabel("GSM timeslot phase [ms]")
ax6.set_ylabel("Power variation")
plt.legend()
#plt.savefig('/home/vereese/Documents/PhD/CASPER2023/casper_presentation/gsm1773sumclean.eps', bbox_inches='tight')
#ax6.set_title("Fold freq = 1733.33 Hz, summed over 1315.76 - 1482.95 MHz (clean band)")


fig7, ax7 = plt.subplots()

ax7.set_xlabel("time ms")
ax7.set_ylabel("power variation H pol")
ax7.set_title("H-pol, fold freq = 1733 Hz, ch 550-750 (clean band)")

y=np.load("/home/vereese/git/phd_data/sk_analysis/2210/sk_rfi_ypol_1024_m1_n1.npy")
fig8, ax8 = plt.subplots()
m = normy1733[:,80:120].sum(axis=1)
ax8.plot((m/max(m)) - 1, label="GSM")
ax8.plot(y/max(y), label="% RFI flagged Y pol")
ax8.set_xlabel("frequency ch")
ax8.set_ylabel("normalised for comparison")
ax8.set_title("")
plt.legend()
plt.show()