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
plt.rc('font', size=11, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=11, labelsize=11)
plt.rc(('xtick', 'ytick'), labelsize=11)
plt.rc('legend', fontsize=11)
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

num_pulses = 32*1413 # 32 processors each processed 1413 pulses
M = ["64", "128", "256"]#, "512", "1024", "2048", "4096", "8192", "16384"]
sk = {} # includes lower and upper thresholds with PFA = 0.267% (calculated using 0.0013499 in script)
sk_pfa2 = {} # includes lower and upper thresholds with PFA = 2% (calculated using 0.1 in script)
sk_low = {} # includes only lower thresholds with PFA = 0
sk_low_pfa2 = {}
sk_low_pfa2_risep = {}
msk = {}
sk_pfa2sig4sig = {}

#I = PI("intensity_2210.npy", initialise=False)
#masked = PI("masked_intensity_2210.npy", initialise=False)
#masked.rfi = np.load(DIR+"masked_sf_2210.npy")

for m in M:
    sk.update({m:PI("sk_intensity_M"+m+"_2210.npy", "sk_summed_flags_M"+m+"_2210.npy")})
    #msk.update({m:PI("MSK_intensity_M"+m+"_2210.npy", "MSK_summed_flags"+m+"_2210.npy")})
    #sk_pfa2.update({m:PI("sk_intensity_pfa2_M"+m+"_2210.npy", "sk_pfa2_summed_flags_M"+m+"_2210.npy")})
    sk_low.update({m:PI("sk_intensity_low_M"+m+"_2210.npy", "sk_low_summed_flags_M"+m+"_2210.npy")})
    sk_low_pfa2.update({m:PI("sk_intensity_low_pfa2_M"+m+"_2210.npy", "sk_low_pfa2_summed_flags_M"+m+"_2210.npy")})
    #sk_low_pfa2_risep.update({m:PI("sk_intensity_risep_low_pfa2_M"+m+"_2210.npy", "sk_risep_low_pfa2_summed_flags_M"+m+"_2210.npy")})
    sk_pfa2sig4sig.update({m:PI("sk_intensity_pfa3sig4sig"+m+"_2210.npy", "sk_pfa3sig4sig_summed_flags_M"+m+"_2210.npy")})
#vt = PI("var_threshold_intensity_2210.npy", "vt_summed_flags_2210.npy")
#vt3sig = PI("var_threshold_3sig_intensity_2210.npy", "vt_3sig_summed_flags_2210.npy")
med = PI("median_smoothed_I4_constant.npy", "sf_median_constant.npy")
#med2 = PI("sk_low_M1024_median_smoothed_I4_custom.npy")
#diff_sk = PI("sk_intensity_pfa2_diff_M16384_2210.npy", "sk_pfa2_summed_flags_diff_M16384_2210.npy")

#I.get_on_pulse()
#I.snr_toa_un()
#I.calc_var()

#masked.get_on_pulse()
#masked.snr_toa_un()
#masked.calc_var()


print("med    : ", med.snr)
#print("diff sk: ", diff_sk.snr)

M = [64, 128, 256]#, 512, 1024, 2048, 4096, 8192, 16384]
snr_sk_1349 = []
snr_sk_2 = []
snr_sk_low = []
snr_sk_low_2 = []
snr_sk_low_2_risep = []
snr_msk = []
snr_pfa2sig4sig = []

toa_un_msk = []
toa_un_sk_2 = []
toa_un_sk_1349 = []
toa_un_sk_low = []
toa_un_sk_low_2 = []
toa_un_sk_low_2_risep = []
toa_un_pfa2sig4sig = []

for m in M:
    #snr_msk.append(msk[str(m)].snr)
    #toa_un_msk.append(msk[str(m)].toa_un)
    #snr_sk_1349.append(sk[str(m)].snr)
    #snr_sk_2.append(sk_pfa2[str(m)].snr)
    snr_sk_low.append(sk_low[str(m)].snr)
    snr_sk_low_2.append(sk_low_pfa2[str(m)].snr)
    snr_pfa2sig4sig.append(sk_pfa2sig4sig[str(m)].snr)
    #snr_sk_low_2_risep.append(sk_low_pfa2_risep[str(m)].snr)
    #toa_un_sk_1349.append(sk[str(m)].toa_un)
    #toa_un_sk_2.append(sk_pfa2[str(m)].toa_un)
    toa_un_sk_low.append(sk_low[str(m)].toa_un)
    toa_un_sk_low_2.append(sk_low_pfa2[str(m)].toa_un)
    toa_un_pfa2sig4sig.append(sk_pfa2sig4sig[str(m)].toa_un)
    #toa_un_sk_low_2_risep.append(sk_low_pfa2_risep[str(m)].toa_un)

plt.figure(1)
#plt.plot(M, masked.snr*np.ones(len(M)), '-o', label="masked", linewidth=2)
#plt.plot(M, vt.snr*np.ones(len(M)), '-o', label=">= 5 $\sigma$", linewidth=2)
#plt.plot(M, snr_sk_1349, '-o', label="SK, PFA=0.27%", linewidth=2)
#plt.plot(M, snr_msk, '-o', label="MSK", linewidth=2)
plt.plot(M, snr_sk_low, '-o', label="SK, low", linewidth=2)
plt.plot(M, snr_sk_low_2, '-o', label="SK, low pfa 2", linewidth=2)
plt.plot(M, snr_pfa2sig4sig, '-o', label="SK, pfa 2 sig 4 sig", linewidth=2)
#plt.plot(M, snr_sk_low_2_risep, '-o', label="SK, low pfa 2 risep", linewidth=2)
#plt.plot(M, I.snr*np.ones(len(M)), '-o', label="None", linewidth=2)
plt.plot(M, med.snr*np.ones(len(M)), '-o', label="median", linewidth=2)
#plt.plot(M, med2.snr*np.ones(len(M)), '-o', label="median sk", linewidth=2)
#plt.plot(M, diff_sk.snr*np.ones(len(M)), '-o', label="diff sk ", linewidth=2)

plt.legend()
plt.grid()
plt.xlim([64, 16384])
#plt.ylim([0,8000])
plt.xlabel("M values")
plt.ylabel("SNR")
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/snr.eps', bbox_inches='tight')

plt.figure(2)
plt.plot(M, med.toa_un*np.ones(len(M)), '-o', label="median", linewidth=2)
#plt.plot(M, toa_un_sk_2, '-o', label="SK, PFA=2%", linewidth=2)
#plt.plot(M, I.toa_un*np.ones(len(M)), '-o', label="None", linewidth=2)
#plt.plot(M, toa_un_sk_1349, '-o', label="SK, PFA=0.27%", linewidth=2)
plt.plot(M, toa_un_sk_low, '-o', label="SK, low", linewidth=2)
plt.plot(M, toa_un_sk_low_2, '-o', label="SK, low pfa2", linewidth=2)
plt.plot(M, toa_un_pfa2sig4sig, '-o', label="SK, pfa 2 sig 4 sig", linewidth=2)
#plt.plot(M, toa_un_sk_low_2_risep, '-o', label="SK, low pfa2 risep", linewidth=2)
#plt.plot(M, vt.toa_un*np.ones(len(M)), '-o', label=">= 5 $\sigma$", linewidth=2)
#plt.plot(M, masked.toa_un*np.ones(len(M)), '-o', label="masked", linewidth=2)
#plt.ylim([0,0.1])
plt.legend()
plt.grid()
plt.xlim([64, 16384])
plt.xlabel("M values")
plt.ylabel("TOA uncertainty [$\mu$s]")
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/toa.eps', bbox_inches='tight')"""

plt.figure(3)
#plt.plot(sk_pfa2["1024"].norm_profile, label="pfa 2")
plt.plot(sk_pfa2sig4sig["64"].norm_profile, label="sk pfa2sig4sig")
#plt.plot(sk["1024"].norm_profile, label="pfa 0.27")
plt.plot(med.norm_profile, label="med")
#plt.plot(vt.norm_profile, label="vt")
#plt.plot(I.norm_profile, label="none")
#plt.plot(masked.norm_profile, label="masked")
plt.legend()

#sk_sf = incoherent_dedisperse(sk["1024"].sf, "2210")
#print(sk["1024"].sf[100:12090:100])
plt.figure(4)
#plt.hist(sk_sf.flatten(), bins=1000)
#plt.imshow(sk_sf, origin="lower",aspect="auto", vmax=2000)
plt.plot(incoherent_dedisperse(sk_pfa2sig4sig["64"].sf, "2210").mean(axis=0)/90432, label="both") #, origin="lower",aspect="auto")
plt.plot(incoherent_dedisperse(sk_low["64"].sf, "2210").mean(axis=0)/90432, label="low") #, origin="lower",aspect="auto")
plt.legend()

plt.figure(5)
plt.plot(sk_low_pfa2["128"].std, label="sk low pfa 2")
plt.plot(med.std, label="med")
plt.legend()

plt.figure(6)
#plt.plot(sk["1024"].rfi, label="SK")
plt.plot(sk_low["128"].rfi, label="SK low")
plt.legend()
plt.show()

'''plt.figure(3)
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
plt.legend()'''




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
#plt.ylabel("normalized pulsar intensity profile")
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/test_profiles.png', bbox_inches='tight')