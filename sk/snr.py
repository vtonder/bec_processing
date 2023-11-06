import time

import matplotlib.ticker
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from constants import time_resolution, J0437_samples_T, pulsars
from pulsar import PulsarIntensity, incoherent_dedisperse

DIR = "/home/vereese/git/phd_data/sk_analysis/2210/"
TAG = "2210"

# Setup fonts and sizes for publication, based on page dimensions in inches
# This is tuned for LaTeX beamer slides
textwidth =  9.6 #128.0 / 25.4 #
textheight = 7 # 96.0 / 25.4
plt.rc('font', size=22, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=22, labelsize=22)
plt.rc(('xtick', 'ytick'), labelsize=22)
plt.rc('legend', fontsize=14)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

#pulsar intensity class
class PI(PulsarIntensity):
    def __init__(self, file_name, sf = None, initialise=True):
        super(PI, self).__init__(DIR, TAG, file_name, sf)

        self.profile = self.I.sum(axis=0)
        self.std = []
        self.norm_profile = self.profile/max(self.profile)
        self.snr = 0
        self.toa_un = 0

        if initialise:
            self.compute()

    def snr_toa_un(self):
        m = np.mean(self.profile[0:self.pulse_width])
        s = np.std(self.profile[0:self.pulse_width])

        for i in np.arange(self.pulse_width):
            self.snr = self.snr + (self.profile[self.pulse_start + i] - m)
        self.snr = self.snr / (s * np.sqrt(self.pulse_width))
        self.toa_un = self.pulse_width * time_resolution / self.snr

        print("SNR         : ", self.snr)
        print("TOA un [us] : ", self.toa_un)

    def compute(self):
        #self.apply_mask()
        t1 = time.time()
        self.get_on_pulse()
        print("Get on pulse took: ", time.time()- t1)
        t1 = time.time()
        self.snr_toa_un()
        print("SNR TOA UNN  took: ", time.time()- t1)

def make_snr_toa_list(sk_dict, M):
    snr = []
    toa = []
    for m in M:
        m = str(m)
        snr.append(sk_dict[m].snr)
        toa.append(sk_dict[m].toa_un)

    return snr, toa


sk = {} # includes lower and upper thresholds with PFA = 0.267% (calculated using 0.0013499 in script)
sk_pfa2 = {} # includes lower and upper thresholds with PFA = 2% (calculated using 0.01 in script)
sk_4sig = {} # includes both thresholds with PFA = 4 sigma
sk_4sig_median = {} # both thresholds at  sigma and median smoothing afterwards
sk_low = {} # includes only lower thresholds with PFA = 0.267% (calculated using 0.0013499 in script)
sk_low_pfa2 = {} # includes only lower thresholds with PFA = 2% (calculated using 0.01 in script)
sk_low_4sig = {} # includes only lower thresholds with PFA = 4 sigma (calculated using 3.1671241833008956e-05 in script)
sk_low_4sig_median = {} # includes only lower thresholds with PFA = 4 sigma which was median filtered afterwards
sk_low_pfa2_risep = {} # only lower thresholds, real and imaginary processed separately - made no difference
sk_pfa2sig4sig = {} # 2 sigma lower threshold 4 sigma upper threshold
sk_pfa3sig4sig = {} # 4 sigma lower threshold 4 sigma upper threshold
sk_pfa3sigsklim = {} # 3 sigma lower threshold sk median upper threshold. sk median taken only in clean RFI channels
sk_pfa4sigsklim = {} # 4 sigma lower thresholds sk median. sk median taken only in clean RFI channels
sk_pfa4sig4sigsklim = {} # PFA = 4 sigma applied to all lower and upper between 260 - 330, the rest of upper is sklim
sk_pfa4sigskmaxlim = {} # 4 sigma lower thresholds sk max upper threshold. sk max taken only in clean RFI channels

# Ran upper and lower thresholds using 2% PFA but lower used M = 16384 and upper M = 512
# made special script: diff_M_sk.py for this experiment
# diff_sk = PI("sk_intensity_pfa2_diff_M16384_2210.npy", "sk_pfa2_summed_flags_diff_M16384_2210.npy")

I = PI("intensity_2210.npy", initialise=False)

masked = PI("masked_intensity_2210.npy", initialise=False)
masked.rfi = np.load(DIR+"masked_sf_2210.npy")

I.get_on_pulse()
I.snr_toa_un()

masked.get_on_pulse()
masked.snr_toa_un()
M = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
for m in M:
    m = str(m)
    sk.update({m:PI("sk_intensity_M"+m+"_2210.npy", "sk_summed_flags_M"+m+"_2210.npy", initialise=True)})
    #sk_low_pfa2.update({m: PI("sk_intensity_low_pfa2_M" + m + "_2210.npy", "sk_low_pfa2_summed_flags_M" + m + "_2210.npy", initialise=True)})
    #sk_low.update({m: PI("sk_intensity_low_M" + m + "_2210.npy", "sk_low_summed_flags_M" + m + "_2210.npy", initialise=True)})
    sk_pfa4sigskmaxlim.update({m:PI("sk_intensity_sig4skmaxlim_M"+m+"_2210.npy","sk_sig4skmaxlim_summed_flags_M"+m+"_2210.npy", initialise=True)})
    sk_4sig.update({m:PI("sk_intensity_pfa4sig_M"+m+"_2210.npy", "sk_pfa4sig_summed_flags_M"+m+"_2210.npy", initialise=True)})
    sk_4sig_median.update({m:PI("sk_pfa4sig_M"+m+"_median_I4.npy", "sk_pfa4sig_M"+m+"_sf_median.npy", initialise=True)})
    sk_low_4sig.update({m:PI("sk_intensity_low_pfa4sig_M" + m + "_2210.npy", "sk_low_pfa4sig_summed_flags_M" + m + "_2210.npy", initialise=True)})
    sk_low_4sig_median.update({m:PI("sk_low_pfa4sig_M" + m + "_median_smoothed_I4.npy", "sk_low_pfa4sig_M" + m + "_sf_median.npy", initialise=True)})

vt = PI("var_threshold_4sig_intensity_2210.npy", "vt_4sig_summed_flags_2210.npy", initialise=True)
med = PI("median_I4.npy", "sf_median.npy", initialise=True)
med.rfi = 100*med.sf.sum(axis=1)/(np.floor(J0437_samples_T)*4)

n_ch = [2, 4, 8, 16]
msk_4siglow_M64m1nx = {"1":PI("sk_intensity_low_pfa4sig_M64_2210.npy", initialise=True)}
msk_4siglow_M128m1nx = {"1":PI("sk_intensity_low_pfa4sig_M128_2210.npy", initialise=True)}
msk_4siglow_M4096m1nx = {"1":PI("sk_intensity_low_pfa4sig_M4096_2210.npy", initialise=True)}
vmsk_4siglow_M64m1nx = {"1":PI("sk_intensity_low_pfa4sig_M64_2210.npy", initialise=True)}
#msk_4siglow_M256m1nx = {"1":PI("sk_intensity_low_pfa4sig_M256_2210.npy")}
#msk64m1nx_median = {"1":PI("sk_low_pfa4sig_M64_median_smoothed_I4.npy")}
#msk128m1nx = {"1":PI("sk_intensity_low_pfa4sig_M128_2210.npy")}
for n in n_ch[1:]:
    n = str(n)
    msk_4siglow_M64m1nx.update({n:PI("MSK_intensity_low_sig4_M64_m1_n" + n + "_2210.npy", initialise=True)})
    msk_4siglow_M128m1nx.update({n:PI("MSK_intensity_low_sig4_M128_m1_n" + n + "_2210.npy", initialise=True)})
    msk_4siglow_M4096m1nx.update({n:PI("MSK_intensity_low_sig4_M4096_m1_n" + n + "_2210.npy", initialise=True)})
    #msk_4siglow_M64m1nx[n].rfi = np.load(DIR + "MSK_rfi_64_m1_n" + n + ".npy")
    vmsk_4siglow_M64m1nx.update({n:PI("VMSK_intensity_low_sig4_M64_m1_n" + n + "_2210.npy", initialise=True)})
    #vmsk_4siglow_M64m1nx[n].rfi = np.load(DIR + "VMSK_rfi_64_m1_n" + n + ".npy")

    #msk64m1nx_median.update({n:PI("msk_low_4sig_M64m1n" + n + "_median_I4.npy")})
    #msk128m1nx.update({n:PI("MSK_intensity_low_sig4_M128_m1_n"+n+"_2210.npy")})
    #msk_4siglow_M256m1nx.update({n:PI("MSK_intensity_low_sig4_M256_m1_n" + n + "_2210.npy")})

snr_sk_4sig, toa_un_sk_4sig = make_snr_toa_list(sk_4sig, M)
snr_sk_4sig_med, toa_un_sk_4sig_med = make_snr_toa_list(sk_4sig_median, M)
snr_sk_low_4sig, toa_un_sk_low_4sig = make_snr_toa_list(sk_low_4sig, M)
snr_4sigskmax, toa_un_4sigskmax = make_snr_toa_list(sk_pfa4sigskmaxlim, M)
snr_sk_low_4sig_med, toa_un_sk_low_4sig_med = make_snr_toa_list(sk_low_4sig_median, M)
#snr_sk_low_2, toa_un_sk_low_2 = make_snr_toa_list(sk_low_pfa2, M)

#snr_sk_1349, toa_un_sk_1349 = make_snr_toa_list(sk, M)
#snr_sk_low, toa_un_sk_low = make_snr_toa_list(sk_low, M)

m64_snr, m64_toa = make_snr_toa_list(msk_4siglow_M64m1nx, n_ch)
vm64_snr, vm64_toa = make_snr_toa_list(vmsk_4siglow_M64m1nx, n_ch)
#m64_med_snr, m64_med_toa = make_snr_toa_list(msk64m1nx_median, n_ch)
m128_snr, m128_toa = make_snr_toa_list(msk_4siglow_M128m1nx, n_ch)
m4096_snr, m4096_toa = make_snr_toa_list(msk_4siglow_M4096m1nx, n_ch)
#m256_snr, m256_toa = make_snr_toa_list(msk_4siglow_M256m1nx, n_ch)

fig, ax = plt.subplots()
ax.semilogx(M, snr_sk_low_4sig_med, '-o', label="SK, PFA: low 4$\sigma$ median", linewidth=2, base=2)
ax.semilogx(M, snr_sk_low_4sig, '-o', label="SK, PFA: low 4$\sigma$", linewidth=2, base=2)
#ax.semilogx(M, snr_sk_low_2, '-o', label="SK, PFA: 2%$", linewidth=2, base=2)
#ax.semilogx(M, snr_sk_low, '-o', label="SK, PFA: low 3$\sigma$", linewidth=2, base=2)
ax.semilogx(M, snr_4sigskmax, '-o', label="SK, PFA: 4$\sigma$ SK max", linewidth=2, base=2)
ax.semilogx(M, snr_sk_4sig, '-o', label="SK, PFA: 4$\sigma$", linewidth=2, base=2)
ax.semilogx(M, snr_sk_4sig_med, '-o', label="SK, PFA: 4$\sigma$ median", linewidth=2, base=2)
ax.hlines(y=med.snr,xmin=M[0],xmax=M[-1], colors="blue", linestyle="--", label="median")
ax.hlines(y=vt.snr,xmin=M[0],xmax=M[-1], colors="green", linestyle="--", label=">= 4$\sigma$")
ax.hlines(y=I.snr,xmin=M[0],xmax=M[-1], colors="red", linestyle="--", label="none")
ax.hlines(y=masked.snr,xmin=M[0],xmax=M[-1], colors="yellow", linestyle="--", label="masked")
ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatter(base=2))
ax.set_ylabel("SNR")
ax.set_xlabel("M")
ax.set_xlim([M[0], M[-1]])
ax.legend(loc=3)
ax.grid()
#plt.savefig('/home/vereese/Documents/PhD/jai-2e/sk_snr.eps', bbox_inches='tight')

fig1, ax1 = plt.subplots()
ax1.semilogx(M, toa_un_sk_4sig, '-o', label="SK, PFA: 4$\sigma$", linewidth=2, base=2)
ax1.semilogx(M, toa_un_4sigskmax, '-o', label="SK, PFA: 4$\sigma$ SK max", linewidth=2, base=2)
ax1.semilogx(M, toa_un_sk_low_4sig_med, '-o', label="SK, PFA: low 4$\sigma$ median", linewidth=2, base=2)
ax1.semilogx(M, toa_un_sk_low_4sig, '-o', label="SK, PFA: low  4$\sigma$", linewidth=2, base=2)
ax1.hlines(y=I.toa_un,xmin=M[0],xmax=M[-1], colors="red", linestyle="--", label="none")
ax1.hlines(y=vt.toa_un,xmin=M[0],xmax=M[-1], colors="green",  linestyle="--", label=">= 5$\sigma$")
ax1.hlines(y=med.toa_un,xmin=M[0],xmax=M[-1], colors="blue", linestyle="--", label="median")
ax1.xaxis.set_major_formatter(matplotlib.ticker.LogFormatter(base=2))
ax1.set_xlabel("M values")
ax1.set_ylabel("TOA uncertainty [$\mu$s]")
ax1.set_xlim([M[0], M[-1]])
ax1.legend()
ax1.grid()
#plt.savefig('/home/vereese/Documents/PhD/jai-2e/sk_toa_un.eps', bbox_inches='tight')
#plt.show()

fig2, ax2 = plt.subplots()
#plt.plot(n_ch, m64_med_snr, '-o', label="M64, m = 1 median", linewidth=2)
ax2.plot(n_ch, m64_snr, '-o', label="msk M64, m = 1", linewidth=2)
ax2.plot(n_ch, m128_snr, '-o', label="msk M128, m = 1", linewidth=2)
ax2.plot(n_ch, m4096_snr, '-o', label="msk M4096, m = 1", linewidth=2)
ax2.plot(n_ch, vm64_snr, '-o', label="vmsk M64, m = 1", linewidth=2)
for Msk in ["128", "256", "512"]:
    ax2.plot(1, sk_low_4sig[Msk].snr, 'o', label="SK, M="+Msk)
#plt.plot(n_ch, m128_snr, '-o', label="M128, m = 1", linewidth=2)
#plt.plot(n_ch, m256_snr, '-o', label="M256, m = 1", linewidth=2)
ax2.set_xlabel("n")
ax2.set_ylabel("SNR")
ax2.grid()
ax2.legend()

phi = np.arange(0,1,1/len(I.profile))
fig3, ax3 = plt.subplots()
ax3.plot(phi, I.norm_profile+0.6, label="none")
ax3.plot(phi, sk_4sig["1024"].norm_profile+0.5, label="SK, PFA: 4$\sigma$")
ax3.plot(phi, vt.norm_profile+0.4, label=">= 4$\sigma$")
ax3.plot(phi, sk_pfa4sigskmaxlim["1024"].norm_profile + 0.3, label="SK, PFA: 4$\sigma$ SK max")
ax3.plot(phi, sk_low_4sig["1024"].norm_profile + 0.2, label="SK, PFA: low 4$\sigma$")
ax3.plot(phi, med.norm_profile+0.1, label="median")
ax3.plot(phi, sk_low_4sig_median["1024"].norm_profile, label="SK, PFA: low 4$\sigma$ median")
ax3.set_ylabel("normalized pulsar intensity profile")
ax3.set_xlabel("pulsar phase")
ax3.set_xlim([0,1])
ax3.grid()
ax3.legend()

var_x = np.load("/home/vereese/git/phd_data/mean_analysis/2210/var_0x_1024.npy")
fig4, ax4 = plt.subplots()
ax4.plot(var_x[:,0]/100, label="variance")
#ax4.plot(sk["2048"].rfi, label="sk pfa 3 sig")
#ax4.plot(sk_low["2048"].rfi, label="sk low 3 sig")
#ax4.plot(sk["4096"].rfi, label="sk pfa 3 sig")
ax4.plot(sk_4sig["128"].rfi, label="sk pfa 4 sig M=512")
ax4.plot(sk_4sig["256"].rfi, label="sk pfa 4 sig M=512")
ax4.plot(sk_4sig["512"].rfi, label="sk pfa 4 sig M=512")
ax4.plot(sk_4sig["1024"].rfi, label="sk pfa 4 sig M=1024")
ax4.plot(sk_4sig["2048"].rfi, label="sk pfa 4 sig M=2048")
ax4.plot(sk_4sig["4096"].rfi, label="sk pfa 4 sig M=4096")
ax4.plot(sk_4sig["8192"].rfi, label="sk pfa 4 sig M=8192")

ax4.plot(sk_low_4sig["4096"].rfi, label="sk low 4 sig M4096")
ax4.plot(med.rfi, label="med")
ax4.plot(masked.rfi, label="masked")
ax4.set_ylabel("% RFI flagged")
ax4.set_xlabel("frequency ")
ax4.legend()
plt.show()

fig5, ax5 = plt.subplots()
ax5.plot(sk_4sig["2048"].I.sum(axis=1), label="sk pfa 4 sig")
ax5.plot(I.I.sum(axis=1), label="none")
ax5.plot(masked.I.sum(axis=1), label="masked")
plt.legend()
plt.show()


#plt.plot(sk_pfa2["1024"].rfi, label="sk pfa 2 %")
#plt.semilogy(sk_low_pfa4sig["64"].rfi, label="sk low 4sig M64")
#plt.semilogy(sk_low_pfa4sig["128"].rfi, label="sk low 4sig M128")
#plt.semilogy(sk_low_pfa4sig["256"].rfi, label="sk low 4sig M256")
#ax4.semilogy(sk_low_pfa4sig["4096"].rfi, label="sk low 4sig M4096")

#plt.plot(sk_low_pfa4sig["256"].rfi, label="sk low 4sig M265")
#plt.plot(sk["1024"].rfi, label="sk 3sig ")
#plt.plot(msk_4siglow_M64m1nx["8"].rfi, label="msk M64m1n8")
#plt.plot(vmsk_4siglow_M64m1nx["8"].rfi, label="msk M64m1n8")
#plt.semilogy(msk_4siglow_M256m1nx["2"].rfi, label="msk M256m1n2")

#plt.semilogy(vmsk_4siglow_M256m1n2.rfi, label="vmsk M256m1n2")
#plt.plot(sk_pfa4sigsklim["1024"].rfi, label="sk 4sig sklim")
#plt.plot(sk_pfa4sig4sigsklim["1024"].rfi, label="sk 4sig 4sig sklim")
#plt.plot(sk_4sig["64"].rfi, label="sk 4sig")
#
#plt.plot(vt.rfi, label="var threshold")

#plt.plot(sk_low_pfa4sig["1024"].rfi, label="sk low pfa 4 sig")
#plt.plot(sk_pfa4sig4sigsklim["1024"].rfi, label="sk pfa 4 sig 4 sig sklim")
#plt.plot(sk_pfa4sigsklim["1024"].rfi, label="sk pfa 4 sig sklim")
#plt.plot(sk_low_pfa2["1024"].rfi, label="sk low pfa 2")
#plt.plot(sk_pfa2sig4sig["128"].rfi, label="SK low")


#sk_sf = incoherent_dedisperse(sk["1024"].sf, "2210")
#print(sk["1024"].sf[100:12090:100])
#plt.figure(4)
#plt.hist(sk_sf.flatten(), bins=1000)
#plt.imshow(sk_sf, origin="lower",aspect="auto", vmax=2000)
#plt.plot(incoherent_dedisperse(sk_pfa2sig4sig["1024"].sf, "2210").mean(axis=0)/90432, label="sk lim") #, origin="lower",aspect="auto")
#plt.plot(incoherent_dedisperse(sk_pfa3sig4sig["1024"].sf, "2210").mean(axis=0)/90432, label="sk pfa 3sig 4sig") #, origin="lower",aspect="auto")
#plt.plot(incoherent_dedisperse(sk_low["1024"].sf, "2210").mean(axis=0)/90432, label="sk low pfa 3sig") #, origin="lower",aspect="auto")
#plt.plot(incoherent_dedisperse(sk["1024"].sf, "2210").mean(axis=0)/90432, label="sk 3sig") #, origin="lower",aspect="auto")
#plt.plot(incoherent_dedisperse(sk_pfa2["1024"].sf, "2210").mean(axis=0)/90432, label="sk pfa 2 %") #, origin="lower",aspect="auto")
#plt.plot(incoherent_dedisperse(sk_4sig["1024"].sf, "2210").mean(axis=0), label="sk 4 sig") #, origin="lower",aspect="auto")
#plt.plot(incoherent_dedisperse(sk_low_pfa4sig["1024"].sf, "2210").sum(axis=0), label="sk low pfa 4 sig") #, origin="lower",aspect="auto")
#plt.plot(incoherent_dedisperse(sk_low_pfa4sig["1024"].sf, "2210").sum(axis=0), label="sk low pfa 4 sig") #, origin="lower",aspect="auto")
#plt.plot(med.sf.sum(axis=0), label="median") #, origin="lower",aspect="auto")

#plt.plot(incoherent_dedisperse(sk_pfa4sigsklim["1024"].sf, "2210").mean(axis=0), label="sk pfa 4sig sklim") #, origin="lower",aspect="auto")
#plt.plot(incoherent_dedisperse(sk_pfa4sig4sigsklim["1024"].sf, "2210").mean(axis=0), label="sk pfa 4sig 4sig sklim") #, origin="lower",aspect="auto")
#plt.plot(incoherent_dedisperse(sk_pfa4sigskmaxlim["1024"].sf, "2210").mean(axis=0), label="sk pfa 4sig sk max") #, origin="lower",aspect="auto")
#plt.plot(incoherent_dedisperse(sk_low_pfa4sig_median["1024"].sf, "2210").mean(axis=0), label="sk low pfa 4sig med") #, origin="lower",aspect="auto")
#plt.legend()

#plt.figure(5)
#plt.plot(sk_low_pfa4sig["1024"].std, label="sk low 4sig ")
#plt.plot(sk_low_pfa4sig_median["1024"].std, label="sk low 4sig median")
#plt.plot(sk_pfa4sigsklim["1024"].std, label="sk 4sig sklim")
#plt.plot(sk_pfa4sig4sigsklim["1024"].std, label="sk 4sig 4sig sklim")
#plt.plot(sk_4sig["1024"].std, label="sk 4sig")
#plt.plot(med.std, label="med")
#plt.legend()


