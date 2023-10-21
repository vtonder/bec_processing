import matplotlib.ticker
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from constants import time_resolution, pulsars
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

    def calc_var(self):
        for i in np.arange(1024):
            if not any(self.I[i,:]):
                continue
            profile = self.I[i,:]/np.mean(self.I[i,:])
            self.std.append(np.sqrt(np.var(profile[0:1000])))

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
        self.apply_mask()
        self.get_on_pulse()
        self.snr_toa_un()
        self.calc_var()

def make_snr_toa_list(sk_dict, M):
    snr = []
    toa = []
    for m in M:
        snr.append(sk_dict[m].snr)
        toa.append(sk_dict[m].toa_un)

    return snr, toa

num_pulses = 32*1413 # 32 processors each processed 1413 pulses
M = ["64", "128", "256", "512", "1024", "2048", "4096", "8192"]
sk = {} # includes lower and upper thresholds with PFA = 0.267% (calculated using 0.0013499 in script)
sk_pfa2 = {} # includes lower and upper thresholds with PFA = 2% (calculated using 0.01 in script)
sk_4sig = {} # includes both thresholds with PFA = 4 sigma
sk_low = {} # includes only lower thresholds with PFA = 0.267% (calculated using 0.0013499 in script)
sk_low_pfa2 = {} # includes only lower thresholds with PFA = 2% (calculated using 0.01 in script)
sk_low_pfa4sig = {} # includes only lower thresholds with PFA = 4 sigma (calculated using 3.1671241833008956e-05 in script)
sk_low_pfa4sig_median = {} # includes only lower thresholds with PFA = 4 sigma which was median filtered afterwards

sk_low_pfa2_risep = {} # only lower thresholds, real and imaginary processed separately - made no difference

sk_pfa2sig4sig = {} # 2 sigma lower threshold 4 sigma upper threshold
sk_pfa3sig4sig = {} # 4 sigma lower threshold 4 sigma upper threshold
sk_pfa3sigsklim = {} # 3 sigma lower threshold sk median upper threshold. sk median taken only in clean RFI channels
sk_pfa4sigsklim = {} # 4 sigma lower thresholds sk median. sk median taken only in clean RFI channels
sk_pfa4sig4sigsklim = {} # PFA = 4 sigma applied to all lower and upper between 260 - 330, the rest of upper is sklim
sk_pfa4sigskmaxlim = {} # 4 sigma lower thresholds sk max upper threshold. sk max taken only in clean RFI channels

I = PI("intensity_2210.npy", initialise=True)
masked = PI("masked_intensity_2210.npy", initialise=False)
masked.rfi = np.load(DIR+"masked_sf_2210.npy")

for m in M:
    sk.update({m:PI("sk_intensity_M"+m+"_2210.npy", "sk_summed_flags_M"+m+"_2210.npy")})
    sk_pfa4sigskmaxlim.update({m:PI("sk_intensity_sig4skmaxlim_M"+m+"_2210.npy","sk_sig4skmaxlim_summed_flags_M"+m+"_2210.npy")})
    sk_4sig.update({m:PI("sk_intensity_pfa4sig_M"+m+"_2210.npy", "sk_pfa4sig_summed_flags_M"+m+"_2210.npy")})
    sk_low_pfa4sig.update({m:PI("sk_intensity_low_pfa4sig_M"+m+"_2210.npy", "sk_low_pfa4sig_summed_flags_M"+m+"_2210.npy")})
    sk_low_pfa4sig_median.update({m:PI("sk_low_pfa4sig_M"+m+"_median_smoothed_I4.npy", "sk_low_pfa4sig_M"+m+"_sf_median.npy")})

    #sk_low.update({m:PI("sk_intensity_low_M"+m+"_2210.npy", "sk_low_summed_flags_M"+m+"_2210.npy")})
    #sk_low_pfa2.update({m:PI("sk_intensity_low_pfa2_M"+m+"_2210.npy", "sk_low_pfa2_summed_flags_M"+m+"_2210.npy")})
    #sk_pfa4sig4sigsklim.update({m:PI("sk_intensity_sig4sig4sklim_M"+m+"_2210.npy", "sk_sig4sig4sklim_summed_flags_M"+m+"_2210.npy")})
    #sk_low_pfa2_risep.update({m:PI("sk_intensity_risep_low_pfa2_M"+m+"_2210.npy", "sk_risep_low_pfa2_summed_flags_M"+m+"_2210.npy")})
    #sk_pfa3sigsklim.update({m:PI("sk_intensity_pfa3sigsklim_M"+m+"_2210.npy", "sk_pfa3sigsklim_summed_flags_M"+m+"_2210.npy")})
    #sk_pfa4sigsklim.update({m:PI("sk_intensity_pfa4sigsklim_M"+m+"_2210.npy", "sk_pfa4sigsklim_summed_flags_M"+m+"_2210.npy")})
    #sk_pfa2sig4sig.update({m:PI("sk_intensity_pfa2sig4sig_M"+m+"_2210.npy", "sk_pfa3sigsklim_summed_flags_M"+m+"_2210.npy")})
    #sk_pfa3sig4sig.update({m:PI("sk_intensity_pfa3sig4sig_M"+m+"_2210.npy", "sk_pfa3sig4sig_summed_flags_M"+m+"_2210.npy")})

vt = PI("var_threshold_5sig_intensity_2210.npy", "vt_5sig_summed_flags_2210.npy")
med = PI("median_I4.npy", "sf_median.npy")

#msk_M64m2n2 = PI("MSK_intensity_sig4_M64_m2_n2_2210.npy")
#msk_M64m1n8 = PI("MSK_intensity_sig4_M64_m1_n8_2210.npy")
#msk_M64m8n1 = PI("MSK_intensity_sig4_M64_m8_n1_2210.npy")

#msk_4sigskmax_M64m2n2 = PI("MSK_intensity_sig4skmax_M64_m2_n2_2210.npy")
#msk_4sigskmax_M64m4n4 = PI("MSK_intensity_sig4skmax_M64_m4_n4_2210.npy")
#msk_4sigskmax_M64m4n2 = PI("MSK_intensity_sig4skmax_M64_m4_n2_2210.npy")
#msk_4sigskmax_M64m2n4 = PI("MSK_intensity_sig4skmax_M64_m2_n4_2210.npy")
#msk_4sigskmax_M64m1n8 = PI("MSK_intensity_sig4skmax_M64_m1_n8_2210.npy")
#msk_4sigskmax_M64m8n1 = PI("MSK_intensity_sig4skmax_M64_m8_n1_2210.npy")
#msk_4sigskmax_M64m16n1 = PI("MSK_intensity_sig4skmax_M64_m16_n1_2210.npy")
#msk_4sigskmax_M64m4n4 = PI("MSK_intensity_sig4skmax_M64_m4_n4_2210.npy")

#msk_4sigskmax_M128m2n2 = PI("MSK_intensity_sig4skmax_M128_m2_n2_2210.npy")
#msk_4sigskmax_M128m8n1 = PI("MSK_intensity_sig4skmax_M128_m8_n1_2210.npy")
#msk_4sigskmax_M256m2n2 = PI("MSK_intensity_sig4skmax_M256_m2_n2_2210.npy")
#mskv_4sigskmax_M256m2n2 = PI("MSKv_intensity_sig4skmax_M256_m2_n2_2210.npy")
#msk_4sigskmax_M256m8n1 = PI("MSK_intensity_sig4skmax_M256_m8_n1_2210.npy")
#n_ch = ["1", "2", "4", "8"]#, "16"]
n_ch = ["1", "2", "8"]#, "16"]
msk_4siglow_M64m1nx = {"1":PI("sk_intensity_low_pfa4sig_M64_2210.npy")}
#msk_4siglow_M256m1nx = {"1":PI("sk_intensity_low_pfa4sig_M256_2210.npy")}
#msk64m1nx_median = {"1":PI("sk_low_pfa4sig_M64_median_smoothed_I4.npy")}
#msk128m1nx = {"1":PI("sk_intensity_low_pfa4sig_M128_2210.npy")}
for m in n_ch[1:]:
    msk_4siglow_M64m1nx.update({m:PI("MSK_intensity_low_sig4_M64_m1_n" + m + "_2210.npy")})
#    msk64m1nx_median.update({m:PI("msk_low_4sig_M64m1n"+m+"_median_I4.npy")})
#    msk128m1nx.update({m:PI("MSK_intensity_low_sig4_M128_m1_n"+m+"_2210.npy")})
     #msk_4siglow_M256m1nx.update({m:PI("MSK_intensity_low_sig4_M256_m1_n" + m + "_2210.npy")})

msk_4siglow_M64m1nx["8"].rfi = np.load(DIR+"MSK_rfi_64_m1_n8.npy")
#msk_4siglow_M256m1nx["2"].rfi = np.load(DIR+"msk_M256m1n2.npy")

#msk_4siglow_M64m2n8 = PI("MSK_intensity_low_sig4_M64_m2_n8_2210.npy")
#vmsk_4siglow_M256m1n2 = PI("VMSK_intensity_low_sig4skmax_M256_m1_n2_2210.npy")
#vmsk_4siglow_M256m1n2.rfi = np.load(DIR+"vmsk_M256m1n2.npy")
#vmsk_4siglow_M64m1n2 = PI("VMSK_intensity_low_sig4_M64_m1_n2_2210.npy")
vmsk_4siglow_M64m1n8 = PI("VMSK_intensity_low_sig4_M64_m1_n8_2210.npy")
vmsk_4siglow_M64m1n8.rfi = np.load(DIR+"VMSK_rfi_64_m1_n8.npy")
#med2 = PI("sk_low_M1024_median_smoothed_I4_custom.npy")
#diff_sk = PI("sk_intensity_pfa2_diff_M16384_2210.npy", "sk_pfa2_summed_flags_diff_M16384_2210.npy")

I.get_on_pulse()
I.snr_toa_un()
I.calc_var()

"""masked.get_on_pulse()
masked.snr_toa_un()
masked.calc_var()"""


print("med    : ", med.snr)
#print("diff sk: ", diff_sk.snr)


#snr_sk_1349, toa_un_sk_1349 = make_snr_toa_list(sk, M)
#snr_sk_2, toa_un_sk_2 = make_snr_toa_list(sk_pfa2, M)
snr_sk_4sig, toa_un_sk_4sig = make_snr_toa_list(sk_4sig, M)

#snr_sk_low, toa_un_sk_low = make_snr_toa_list(sk_low, M)
#snr_sk_low_2, toa_un_sk_low_2 = make_snr_toa_list(sk_low_pfa2, M)
snr_sk_low_4sig, toa_un_sk_low_4sig = make_snr_toa_list(sk_low_pfa4sig, M)
#snr_sk_low_2_risep, toa_un_sk_low_2_risep = make_snr_toa_list(sk_low_pfa2_risep, M)

#snr_msk, toa_un_msk = make_snr_toa_list(msk, M)

#snr_pfa2sig4sig, toa_un_pfa2sig4sig = make_snr_toa_list(sk_pfa2sig4sig, M)
#snr_pfa3sig4sig, toa_un_pfa3sig4sig = make_snr_toa_list(sk_pfa3sig4sig, M)
#snr_pfa3sigsklim, toa_un_pfa3sigsklim = make_snr_toa_list(sk_pfa3sigsklim, M)
#snr_pfa4sigsklim, toa_un_pfa4sigsklim = make_snr_toa_list(sk_pfa4sigsklim, M)
#snr_pfa4sig4sigsklim, toa_un_pfa4sig4sigsklim = make_snr_toa_list(sk_pfa4sig4sigsklim, M)
snr_4sigskmax, toa_un_4sigskmax = make_snr_toa_list(sk_pfa4sigskmaxlim, M)
snr_sk_low_4sig_med, toa_un_sk_low_4sig_med = make_snr_toa_list(sk_low_pfa4sig_median,M)

M = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
fig, ax = plt.subplots()
ax.semilogx(M, snr_sk_low_4sig_med, '-o', label="SK, PFA: low 4$\sigma$ median", linewidth=2, base=2)
ax.semilogx(M, snr_sk_low_4sig, '-o', label="SK, PFA: low 4$\sigma$", linewidth=2, base=2)
ax.semilogx(M, snr_4sigskmax, '-o', label="SK, PFA: 4$\sigma$ SK max", linewidth=2, base=2)
ax.semilogx(M, snr_sk_4sig, '-o', label="SK, PFA: 4$\sigma$", linewidth=2, base=2)
ax.hlines(y=med.snr,xmin=M[0],xmax=M[-1], colors="blue", linestyle="--", label="median")
ax.hlines(y=vt.snr,xmin=M[0],xmax=M[-1], colors="green", linestyle="--", label=">= 5$\sigma$")
ax.hlines(y=I.snr,xmin=M[0],xmax=M[-1], colors="red", linestyle="--", label="none")
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
plt.show()

m64_snr, m64_toa = make_snr_toa_list(msk_4siglow_M64m1nx, n_ch)
#m64_med_snr, m64_med_toa = make_snr_toa_list(msk64m1nx_median, n_ch)
#m128_snr, m128_toa = make_snr_toa_list(msk128m1nx, n_ch)
#m256_snr, m256_toa = make_snr_toa_list(msk_4siglow_M256m1nx, n_ch)
n_ch = [1,2,8]

plt.figure(2)
#plt.plot(n_ch, m64_med_snr, '-o', label="M64, m = 1 median", linewidth=2)
plt.plot(n_ch, m64_snr, '-o', label="msk M64, m = 1", linewidth=2)
#plt.plot(n_ch, m128_snr, '-o', label="M128, m = 1", linewidth=2)
#plt.plot(n_ch, m256_snr, '-o', label="M256, m = 1", linewidth=2)
#plt.plot(2, vmsk_4siglow_M64m1n2.snr, 'o', label="vmsk M64 m = 1")
plt.plot(8, vmsk_4siglow_M64m1n8.snr, 'o', label="vmsk M64 m = 1")
#plt.plot(8, msk_4siglow_M64m1n8.snr, 'o', label="msk M64 m = 1")
#plt.plot(2, vmsk_4siglow_M256m1n2 .snr, 'o', label="vmsk M256 m = 1")
plt.xlabel("n")
plt.ylabel("SNR")
plt.grid()
plt.legend()


#plt.figure(1)
#plt.plot(M, masked.snr*np.ones(len(M)), '-o', label="masked", linewidth=2)
#plt.plot(M, vt.snr*np.ones(len(M)), '-o', label=">= 5 $\sigma$", linewidth=2)
#plt.plot(M, snr_sk_1349, '-o', label="SK, PFA=0.27%", linewidth=2)
#plt.plot(M, snr_msk, '-o', label="MSK", linewidth=2)
#plt.plot(M, snr_sk_low, '-o', label="SK, low", linewidth=2)
#plt.plot(M, snr_sk_low_2, '-o', label="SK, low pfa 2", linewidth=2)
#plt.plot(M, snr_sk_2, '-o', label="SK, pfa 2", linewidth=2)
#plt.plot(M, snr_sk_low_4sig_med, '-o', label="SK, low 4 sig median", linewidth=2)
#plt.plot(M, snr_sk_low_4sig, '-o', label="SK, low pfa 4 sig", linewidth=2)
#plt.plot(M, snr_4sigskmax, '-o', label="SK, pfa 4 sig sk max lim", linewidth=2)
#plt.plot(M, snr_sk_4sig, '-o', label="SK, pfa 4 sig", linewidth=2)
#plt.plot(M, snr_pfa2sig4sig, '-o', label="SK, pfa 2 sig 4 sig", linewidth=2)
#plt.plot(M, snr_pfa3sig4sig, '-o', label="SK, pfa 3 sig 4 sig", linewidth=2)
#plt.plot(M, snr_pfa3sigsklim, '-o', label="SK, pfa 3 sig sk lim", linewidth=2)
#plt.plot(M, snr_pfa4sigsklim, '-o', label="SK, pfa 4 sig sk lim", linewidth=2)
#plt.plot(M, snr_pfa4sig4sigsklim, '-o', label="SK, pfa 4 sig 4 sig sk lim", linewidth=2)

#plt.plot(M, snr_sk_low_2_risep, '-o', label="SK, low pfa 2 risep", linewidth=2)
#plt.plot(M, I.snr*np.ones(len(M)), '-o', label="None", linewidth=2)
#plt.plot(M, med.snr*np.ones(len(M)), '-o', label="median", linewidth=2)
#plt.plot(M, msk_M64m2n2.snr*np.ones(len(M)), '-o', label="msk M64m2n2", linewidth=2)
#plt.plot([128], msk_4sigskmax_M64m2n2.snr, 'o', label="msk sig4skmax M64m2n2", linewidth=2)
#plt.plot([256], msk_4sigskmax_M128m2n2.snr, 'o', label="msk sig4skmax M128m2n2", linewidth=2)
#plt.plot([128], msk_4sigskmax_M128m8n1.snr, 'o', label="msk sig4skmax M128m8n1", linewidth=2)
#plt.plot([512], msk_4sigskmax_M256m2n2.snr, 'o', label="msk sig4skmax M256m2n2", linewidth=2)
#plt.plot([512], mskv_4sigskmax_M256m2n2.snr, 'o', label="mskv sig4skmax M256m2n2", linewidth=2)
#plt.plot([256], msk_4sigskmax_M256m8n1.snr, 'o', label="msk sig4skmax M256m8n1", linewidth=2)
#plt.plot([256], msk_4sigskmax_M64m4n4.snr, 'o', label="msk sig4skmax M64m4n4", linewidth=2)
#plt.plot([128], msk_4sigskmax_M64m4n2.snr, 'o', label="msk sig4skmax M64m4n2", linewidth=2)
#plt.plot([256], msk_4sigskmax_M64m2n4.snr, 'o', label="msk sig4skmax M64m2n4", linewidth=2)
#plt.plot([512], msk_4sigskmax_M64m1n8.snr, 'o', label="msk sig4skmax M64m1n8", linewidth=2)
#plt.plot([64], msk_4sigskmax_M64m1n8.snr, 'o', label="msk sig4skmax M64m1n8", linewidth=2)
#plt.plot([64], msk_4sigskmax_M64m16n1.snr, 'o', label="msk sig4skmax M64m16n1", linewidth=2)
#plt.plot(M, msk_M64m1n8.snr*np.ones(len(M)), '-o', label="msk sig4 M64m1n8", linewidth=2)
#plt.plot(M, msk_M64m8n1.snr*np.ones(len(M)), '-o', label="msk sig4 M64m8n1", linewidth=2)
#plt.plot([64], msk64m1nx["2"].snr, 'o', label="msk sig4 low M64m1n2", linewidth=2)
#plt.plot([64], msk64m1nx["4"].snr, 'o', label="msk sig4 low M64m1n4", linewidth=2)
#plt.plot([64], msk64m1nx["8"].snr, 'o', label="msk sig4 low M64m1n8", linewidth=2)
#plt.plot([128], msk_4siglow_M64m2n8.snr, 'o', label="msk sig4 low M64m2n8", linewidth=2)
#plt.plot([64], msk_4siglow_M64m1n16.snr, 'o', label="msk sig4 low M64m1n16", linewidth=2)

#plt.plot(M, med2.snr*np.ones(len(M)), '-o', label="median sk", linewidth=2)
#plt.plot(M, diff_sk.snr*np.ones(len(M)), '-o', label="diff sk ", linewidth=2)
#plt.xlim([64, 16384])
#plt.ylim([0,8000])
#plt.xlabel("M values")
#plt.ylabel("SNR")
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/snr.eps', bbox_inches='tight')


#plt.figure(2)
#plt.plot(M, med.toa_un*np.ones(len(M)), '-o', label="median", linewidth=2)
#plt.plot(M, toa_un_sk_2, '-o', label="SK, PFA=2%", linewidth=2)
#plt.plot(M, I.toa_un*np.ones(len(M)), '-o', label="None", linewidth=2)
#plt.plot(M, toa_un_sk_1349, '-o', label="SK, PFA=0.27%", linewidth=2)
#plt.plot(M, toa_un_sk_low, '-o', label="SK, low", linewidth=2)
#plt.plot(M, toa_un_sk_low_2, '-o', label="SK, low pfa2", linewidth=2)
#plt.plot(M, toa_un_sk_low_4sig, '-o', label="SK, low pfa 4sig", linewidth=2)
#plt.plot(M, toa_un_sk_4sig, '-o', label="SK, pfa 4sig", linewidth=2)
#plt.plot(M, toa_un_pfa2sig4sig, '-o', label="SK, pfa 2 sig 4 sig", linewidth=2)
#plt.plot(M, toa_un_pfa3sig4sig, '-o', label="SK, pfa 3 sig 4 sig", linewidth=2)
#plt.plot(M, toa_un_pfa3sigsklim, '-o', label="SK, pfa 3 sig sk lim", linewidth=2)
#plt.plot(M, toa_un_pfa4sigsklim, '-o', label="SK, pfa 4 sig sk lim", linewidth=2)
#plt.plot(M, toa_un_pfa4sig4sigsklim, '-o', label="SK, pfa 4 sig 4 sig sk lim", linewidth=2)
#plt.plot(M, toa_un_sk_low_2_risep, '-o', label="SK, low pfa2 risep", linewidth=2)
#plt.plot(M, vt.toa_un*np.ones(len(M)), '-o', label=">= 5 $\sigma$", linewidth=2)
#plt.plot(M, masked.toa_un*np.ones(len(M)), '-o', label="masked", linewidth=2)
#plt.ylim([0,0.1])
#plt.legend()
#plt.grid()
#
#plt.xlabel("M values")
#plt.ylabel("TOA uncertainty [$\mu$s]")
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/toa.eps', bbox_inches='tight')"""

plt.figure(3)
plt.plot(I.norm_profile+0.6, label="none")
plt.plot(sk_4sig["1024"].norm_profile+0.5, label="SK, PFA: 4$\sigma$")
plt.plot(vt.norm_profile+0.4, label=">= 5$\sigma$")
plt.plot(sk_pfa4sigskmaxlim["1024"].norm_profile + 0.3, label="SK, PFA: 4$\sigma$ SK max")
plt.plot(sk_low_pfa4sig["1024"].norm_profile+0.2, label="SK, PFA: low 4$\sigma$")
plt.plot(med.norm_profile+0.1, label="median")
plt.plot(sk_low_pfa4sig_median["1024"].norm_profile, label="SK, PFA: low 4$\sigma$ median")
plt.xlabel("pulsar phase")
#plt.xlim([0,1])
plt.ylabel("normalized pulsar intensity profile")
plt.grid()
plt.legend()
#plt.plot(sk_low_pfa2["4096"].norm_profile, label="sk low pfa 2 %")
#plt.plot(sk_pfa2["4096"].norm_profile, label="sk pfa 2 %")
#plt.plot(sk_pfa3sigsklim["4096"].norm_profile, label="sk pfa 3 sig sk lim")
#plt.plot(sk_pfa4sig4sigsklim["1024"].norm_profile, label="sk pfa 4 sig 4 sig sk lim")
#plt.plot(sk_pfa4sigsklim["1024"].norm_profile, label="sk pfa 4 sig sk lim")
#plt.plot(sk_pfa3sig4sig["4096"].norm_profile, label="sk pfa2sig4sig")
#plt.plot(sk["4096"].norm_profile, label="pfa 0.27")
#plt.plot(msk["4096"].norm_profile, label="msk")
#plt.plot(msk_M64m2n2.norm_profile, label="msk M64m2n2")
#plt.plot(msk_4sigskmax_M256m2n2.norm_profile, label="msk 4sig skmax M256m2n2")
#plt.plot(mskv_4sigskmax_M256m2n2.norm_profile, label="mskv 4sig skmax M256m2n2")
#plt.plot(msk_M64m1n8.norm_profile, label="msk M64m1n8")
#plt.plot(masked.norm_profile, label="masked")


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

plt.figure(4)
#plt.plot(sk["1024"].rfi, label="sk pfa 3 sig")
#plt.plot(sk_pfa2["1024"].rfi, label="sk pfa 2 %")
#plt.semilogy(sk_low_pfa4sig["64"].rfi, label="sk low 4sig M64")
#plt.semilogy(sk_low_pfa4sig["128"].rfi, label="sk low 4sig M128")
#plt.semilogy(sk_low_pfa4sig["256"].rfi, label="sk low 4sig M256")
#plt.semilogy(sk_low_pfa4sig["4096"].rfi, label="sk low 4sig M4096")
#plt.plot(sk_low_pfa4sig["256"].rfi, label="sk low 4sig M265")
#plt.plot(sk["1024"].rfi, label="sk 3sig ")
plt.plot(msk_4siglow_M64m1nx["8"].rfi, label="msk M64m1n8")
#plt.semilogy(msk_4siglow_M256m1nx["2"].rfi, label="msk M256m1n2")
plt.plot(vmsk_4siglow_M64m1n8.rfi, label="vmsk M64m1n8")
#plt.semilogy(vmsk_4siglow_M256m1n2.rfi, label="vmsk M256m1n2")
#plt.plot(sk_pfa4sigsklim["1024"].rfi, label="sk 4sig sklim")
#plt.plot(sk_pfa4sig4sigsklim["1024"].rfi, label="sk 4sig 4sig sklim")
#plt.plot(sk_4sig["64"].rfi, label="sk 4sig")
#plt.plot(med.rfi, label="med")
#plt.plot(vt.rfi, label="var threshold")
#plt.plot(masked.rfi, label="masked")
#plt.plot(sk_low_pfa4sig["1024"].rfi, label="sk low pfa 4 sig")
#plt.plot(sk_pfa4sig4sigsklim["1024"].rfi, label="sk pfa 4 sig 4 sig sklim")
#plt.plot(sk_pfa4sigsklim["1024"].rfi, label="sk pfa 4 sig sklim")
#plt.plot(sk_low_pfa2["1024"].rfi, label="sk low pfa 2")
#plt.plot(sk_pfa2sig4sig["128"].rfi, label="SK low")
plt.legend()
plt.show()

"""plt.figure(3)
plt.plot(none.I[280,:], label="none")
plt.plot(sk_M16384.I[280,:], label="pfa=0.27")
plt.plot(sk_pfa2_M16384.I[280,:], label="pfa=2")
#plt.plot(med.I[280,:], label="median")
plt.plot(sigma.I[280,:], label="5sigma")
plt.legend()

print(np.mean(none.std)) #0.003348
print(np.mean(masked.std)) #0.003327
print(np.mean(sk_M16384.std)) #0.003304 -- TODO: is it ok that this is so low?

plt.figure(4)
plt.plot(none.std, label="none")
plt.plot(masked.std, label="mask")
plt.plot(sk_M16384.std,label="M=16")
plt.legend()"""




#p1 = (profile["intensity_2210.npy"]-0.8*10**14)/max(profile["intensity_2210.npy"]-0.8*10**14)
#p2 = (profiles_sk_1349["sk_intensity_M16384_2210.npy"]-0.32*10**14)/max(profiles_sk_1349["sk_intensity_M16384_2210.npy"]-0.32*10**14)
#p3 = (profiles_sk_02699["itegrated_sk_pfa_1_intensity_M16384_2210.npy"]-0.31*10**14)/max(profiles_sk_02699["itegrated_sk_pfa_1_intensity_M16384_2210.npy"]-0.31*10**14)
#p4 = (profiles_5sigma["var_threshold_intensity_2210.npy"] - 0.46*10**14)/max(profiles_5sigma["var_threshold_intensity_2210.npy"] - 0.46*10**14)
#p5 = (profile_median["median_smoothed_I4.npy"]-0.29*10**14)/max(profile_median["median_smoothed_I4.npy"]-0.29*10**14)
#phi = np.arange(0,1,1/len(none.profile))
#plt.figure(0)
#plt.plot(phi, p3+1.9, label="PFA = 2%")
#plt.plot(phi, p2+1.5, label="PFA = 0.27%")
#plt.plot(phi, p5+0.9, label="median")
#plt.plot(phi, p4+0.5, label=">= 5$\sigma$")
##plt.plot(phi, p1+0.1, label="None")
#plt.plot(none.norm_profile, label="none")
#plt.plot(sk_M16384.norm_profile, label="pfa=0.27")
#plt.plot(sk_pfa2_M16384.norm_profile, label="pfa=2")
##plt.plot(med.norm_profile, label="median")
#plt.plot(sigma.norm_profile, label="5sigma")
#plt.grid()
#plt.legend()
#plt.ylim([0,3])
#plt.xlabel("pulsar phase")
#plt.ylabel("normalized pulsar intensity profile")'''
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/test_profiles.png', bbox_inches='tight')