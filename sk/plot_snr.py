import matplotlib.ticker
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from constants import J0437_samples_T, num_ch, frequencies, thesis_font, a4_textwidth, a4_textheight
from pulsar_snr import PI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", dest = "dir", help = "directory where data is located. default location: /home/vereese/data/phd_data/sk_analysis/2210", default = "/home/vereese/data/phd_data/sk_analysis/2210/")
# "/home/vereese/git/phd_data/sk_analysis/2210/4sig/"
args = parser.parse_args()
DIR = args.dir

# Setup fonts and sizes for publication, based on page dimensions in inches
textwidth = a4_textwidth
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

def make_snr_toa_list(sk_dict, M):
    snr = []
    toa = []
    for m in M:
        m = str(m)
        snr.append(sk_dict[m].snr)
        toa.append(sk_dict[m].toa_un)

    return snr, toa

if __name__ == "__main__":
    sk = {} # includes lower and upper thresholds with PFA = 0.267% (calculated using 0.0013499 in script)
    sk_pfa2 = {} # includes lower and upper thresholds with PFA = 2% (calculated using 0.01 in script)
    sk_4sig = {} # includes both thresholds with PFA = 4 sigma
    sk_4sig_median = {} # both thresholds at sigma and median smoothing afterwards
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
    sk_l1siguskmax = {} # 1 sigma lower thresholds and sk max upper threshold.

    # Ran upper and lower thresholds using 2% PFA but lower used M = 16384 and upper M = 512
    # made special script: diff_M_sk.py for this experiment
    # diff_sk = PI("sk_intensity_pfa2_diff_M16384_2210_p45216.npy", "sk_pfa2_summed_flags_diff_M16384_2210_p45216.npy")

    # replaced all 0s with gausian noise
    #sk_nz4sig =  PI("sk_intensity_l4sigu4sig_M2048_2210_p45216.npy", "sk_summed_flags_l4sigu4sig_M2048_2210_p45216.npy")

    # keep 0s and when flagged 0 the data
    #sk_z4sig =  PI("zsk_intensity_l4sigu4sig_M2048_2210_p45216.npy", "zsk_summed_flags_l4sigu4sig_M2048_2210_p45216.npy")
    #sk_z4sigl = PI("zsk_intensity_l4sig_M2048_2210_p45216.npy", "zsk_summed_flags_l4sig_M2048_2210_p45216.npy")
    #sk_zskmax = PI("zsk_intensity_l4siguskmax_M2048_2210_p45216.npy","zsk_summed_flags_l4siguskmax_M2048_2210_p45216.npy")

    # No RFI mitigation data set
    I = PI(DIR, "intensity_z_2210_p45216.npy", "num_nz_z_2210_p45216.npy", initialise=False)
    I.compute()

    # Only static mask applied
    Im = PI(DIR, "intensity_z_2210_p45216.npy", "num_nz_z_2210_p45216.npy")

    M = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for m in M:
        m = str(m)
        #sk.update({m:PI(DIR, "sk_intensity_M"+m+"_2210_p45216.npy", "sk_summed_flags_M"+m+"_2210_p45216.npy", initialise=True)})
        #sk_low_pfa2.update({m: PI(DIR, "sk_intensity_low_pfa2_M" + m + "_2210_p45216.npy", "sk_low_pfa2_summed_flags_M" + m + "_2210_p45216.npy", initialise=True)})
        #sk_low.update({m: PI(DIR, "sk_intensity_low_M" + m + "_2210_p45216.npy", "sk_low_summed_flags_M" + m + "_2210_p45216.npy", initialise=True)})
        #sk_pfa4sigskmaxlim.update({m:PI(DIR, "sk_intensity_sig4skmaxlim_M"+m+"_2210_p45216.npy","sk_sig4skmaxlim_summed_flags_M"+m+"_2210_p45216.npy", initialise=True)})
        #sk_4sig_median.update({m:PI(DIR, "sk_pfa4sig_M"+m+"_median_I4_2210_p4812.npy", "sk_pfa4sig_M"+m+"_sf_median_2210_p4812.npy", initialise=True)})
        #sk_low_4sig.update({m:PI(DIR, "sk_intensity_l4sig_M" + m + "_m1_n1_2210_p45216.npy", "sk_num_nz_l4sig_M" + m + "_m1_n1_2210_p45216.npy", "sk_summed_flags_l4sig_M" + m + "_m1_n1_2210_p45216.npy", initialise=True)})
        #sk_low_4sig_median.update({m:PI(DIR, "median_z_r4_sk_l4sig_M" + m + "_2210_p4812.npy", "nz_median_z_r4_sk_l4sig_M" + m + "_2210_p4812.npy", "sf_median_z_r4_sk_l4sig_M" + m + "_2210_p4812.npy", initialise=True)})

        sk_4sig.update({m:PI(DIR, "sk_intensity_z_l4sigu4sig_M" + m + "_m1_n1_2210_p45216.npy", "sk_num_nz_z_l4sigu4sig_M" + m + "_m1_n1_2210_p45216.npy", "sk_summed_flags_z_l4sigu4sig_M" + m + "_m1_n1_2210_p45216.npy")})
        sk_l1siguskmax.update({m:PI(DIR, "sk_intensity_z_l1siguskmax_M" + m + "_m1_n1_2210_p45216.npy", "sk_num_nz_z_l1siguskmax_M" + m + "_m1_n1_2210_p45216.npy", "sk_summed_flags_z_l1siguskmax_M" + m + "_m1_n1_2210_p45216.npy")})

    pt = PI(DIR, "pt_intensity_z_2210_p45216.npy", "pt_num_nz_z_2210_p45216.npy", "pt_summed_flags_z_2210_p45216.npy")
    med = PI(DIR, "median_z_r4_2210_p4812.npy", "num_nz_median_z_2210_p4812.npy", "sf_median_z_2210_p4812.npy")
    # median gets run 4 times, therefore it's % RFI is different to the other data sets
    med.rfi_freq = 100 * med.sf.sum(axis = 1) / (np.floor(J0437_samples_T) * 4)
    med.rfi_pulse = 100 * med.sf.sum(axis = 0) / (num_ch * 4)

    snr_sk_4sig, toa_un_sk_4sig = make_snr_toa_list(sk_4sig, M)
    snr_1sigskmax, toa_un_1sigskmax = make_snr_toa_list(sk_l1siguskmax, M)

    #snr_sk_4sig_med, toa_un_sk_4sig_med = make_snr_toa_list(sk_4sig_median, M)
    #snr_sk_low_4sig, toa_un_sk_low_4sig = make_snr_toa_list(sk_low_4sig, M)
    #snr_4sigskmax, toa_un_4sigskmax = make_snr_toa_list(sk_pfa4sigskmaxlim, M)
    #snr_sk_low_4sig_med, toa_un_sk_low_4sig_med = make_snr_toa_list(sk_low_4sig_median, M)
    #snr_sk_low_2, toa_un_sk_low_2 = make_snr_toa_list(sk_low_pfa2, M)
    #snr_sk_1349, toa_un_sk_1349 = make_snr_toa_list(sk, M)
    #snr_sk_low, toa_un_sk_low = make_snr_toa_list(sk_low, M)

    n_ch = [2, 4, 8, 16]
    msk_l1siguskamx_M512m1nx = {}
    vmsk_l1siguskamx_M512m1nx = {}

    #msk_4siglow_M64m1nx = {}
    #msk_4siglow_M128m1nx = {}
    #msk_4siglow_M4096m1nx = {}
    #vmsk_4siglow_M64m1nx = {}
    #msk_4siglow_M256m1nx = {"1":PI("sk_intensity_low_pfa4sig_M256_2210_p45216.npy")}
    #msk64m1nx_median = {"1":PI("sk_low_pfa4sig_M64_median_smoothed_I4.npy")}
    #msk128m1nx = {"1":PI("sk_intensity_low_pfa4sig_M128_2210_p45216.npy")}

    for n in n_ch:
        n = str(n)
        msk_l1siguskamx_M512m1nx.update({n:PI(DIR, "msk_intensity_z_l1siguskmax_M512_m1_n" + n + "_2210_p45216.npy", "msk_num_nz_z_l1siguskmax_M512_m1_n" + n + "_2210_p45216.npy", "msk_summed_flags_z_l1siguskmax_M512_m1_n" + n + "_2210_p45216.npy")})
        vmsk_l1siguskamx_M512m1nx.update({n:PI(DIR, "vmsk_intensity_z_l1siguskmax_M512_m1_n" + n + "_2210_p45216.npy", "vmsk_num_nz_z_l1siguskmax_M512_m1_n" + n + "_2210_p45216.npy", "vmsk_summed_flags_z_l1siguskmax_M512_m1_n" + n + "_2210_p45216.npy")}) 
        #msk_4siglow_M64m1nx.update({n:PI(DIR, "MSK_intensity_low_sig4_M64_m1_n" + n + "_2210_p45216.npy", initialise=True)})
        #msk_4siglow_M128m1nx.update({n:PI(DIR, "MSK_intensity_low_sig4_M128_m1_n" + n + "_2210_p45216.npy", initialise=True)})
        #msk_4siglow_M4096m1nx.update({n:PI(DIR, "MSK_intensity_low_sig4_M4096_m1_n" + n + "_2210_p45216.npy", initialise=True)})
        #msk_4siglow_M64m1nx[n].rfi = np.load(DIR + "MSK_rfi_64_m1_n" + n + ".npy")
        #vmsk_4siglow_M64m1nx.update({n:PI(DIR, "VMSK_intensity_low_sig4_M64_m1_n" + n + "_2210_p45216.npy", initialise=True)})
        #vmsk_4siglow_M64m1nx[n].rfi = np.load(DIR + "VMSK_rfi_64_m1_n" + n + ".npy")
        #msk64m1nx_median.update({n:PI(DIR, "msk_low_4sig_M64m1n" + n + "_median_I4.npy")})
        #msk128m1nx.update({n:PI(DIR, "MSK_intensity_low_sig4_M128_m1_n"+n+"_2210_p45216.npy")})
        #msk_4siglow_M256m1nx.update({n:PI(DIR, "MSK_intensity_low_sig4_M256_m1_n" + n + "_2210_p45216.npy")})

    m512_snr, m512_toa = make_snr_toa_list(msk_l1siguskamx_M512m1nx, n_ch)
    vm512_snr, vm512_toa = make_snr_toa_list(vmsk_l1siguskamx_M512m1nx, n_ch)

    #m64_snr, m64_toa = make_snr_toa_list(msk_4siglow_M64m1nx, n_ch)
    #vm64_snr, vm64_toa = make_snr_toa_list(vmsk_4siglow_M64m1nx, n_ch)
    #m64_med_snr, m64_med_toa = make_snr_toa_list(msk64m1nx_median, n_ch)
    #m128_snr, m128_toa = make_snr_toa_list(msk_4siglow_M128m1nx, n_ch)
    #m4096_snr, m4096_toa = make_snr_toa_list(msk_4siglow_M4096m1nx, n_ch)
    #m256_snr, m256_toa = make_snr_toa_list(msk_4siglow_M256m1nx, n_ch)

    fig, ax = plt.subplots()
    ax.semilogx(M, snr_sk_4sig, '-o', label="SK, PFA: 4$\sigma$", linewidth=2, base=2)
    ax.semilogx(M, snr_1sigskmax, '-o', label="SK, PFA: 1$\sigma$, $SK_{max}$", linewidth=2, base=2)
    ax.hlines(y = med.snr, xmin = M[0], xmax = M[-1], colors="blue", linestyle="--", label = "median")
    ax.hlines(y = pt.snr, xmin = M[0], xmax = M[-1], colors="green", linestyle="--", label =">= 4$\sigma$")
    ax.hlines(y = Im.snr, xmin = M[0], xmax = M[-1], colors="cyan", linestyle="--", label = "static mask")
    ax.hlines(y = I.snr, xmin = M[0], xmax = M[-1], colors="red", linestyle="--", label = "none")
    ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatter(base=2))
    ax.set_ylabel("SNR")
    ax.set_xlabel("M values")
    ax.set_xlim([M[0], M[-1]])
    ax.legend(loc=3)
    ax.grid()
    plt.savefig('/home/vereese/thesis_pics/sk_snr.eps', transparent=True, bbox_inches='tight')

    fig1, ax1 = plt.subplots()
    ax1.semilogx(M, toa_un_sk_4sig, '-o', label="SK, PFA: 4$\sigma$", linewidth=2, base=2)
    ax1.semilogx(M, toa_un_1sigskmax, '-o', label="SK, PFA: 1$\sigma$, $SK_{max}$", linewidth=2, base=2)
    ax1.hlines(y=I.toa_un, xmin=M[0], xmax=M[-1], colors="red", linestyle="--", label="none")
    ax1.hlines(y=Im.toa_un, xmin=M[0], xmax=M[-1], colors="cyan", linestyle="--", label="static mask")
    ax1.hlines(y=pt.toa_un, xmin=M[0], xmax=M[-1], colors="green", linestyle="--", label=">= 4$\sigma$")
    ax1.hlines(y=med.toa_un,xmin=M[0], xmax=M[-1], colors="blue", linestyle="--", label="median")
    ax1.xaxis.set_major_formatter(matplotlib.ticker.LogFormatter(base=2))
    ax1.set_xlabel("M values")
    ax1.set_ylabel("TOA uncertainty [$\mu$s]")
    ax1.set_xlim([M[0], M[-1]])
    ax1.legend()
    ax1.grid()
    plt.savefig('/home/vereese/thesis_pics/sk_toa_un.eps', transparent=True, bbox_inches='tight')

    fig2, ax2 = plt.subplots()
    ax2.plot(n_ch, m512_snr, '-o', label="MSK, M = 512, m = 1", linewidth=2)
    ax2.plot(n_ch, vm512_snr, '-o', label="VMSK, M = 512, m = 1", linewidth=2)
    #ax2.plot(n_ch, m64_snr, '-o', label="msk M64, m = 1", linewidth=2)
    #ax2.plot(n_ch, m128_snr, '-o', label="msk M128, m = 1", linewidth=2)
    #ax2.plot(n_ch, m4096_snr, '-o', label="msk M4096, m = 1", linewidth=2)
    #ax2.plot(n_ch, vm64_snr, '-o', label="vmsk M64, m = 1", linewidth=2)
    markers = ["d", "v", "s", "*"] #, "^", "d"]
    for i, Msk in enumerate(["1024", "2048", "4096", "8192"]):
        ax2.plot(1, sk_l1siguskmax[Msk].snr, markers[i], label="SK, M = "+Msk)
    #plt.plot(n_ch, m128_snr, '-o', label="M128, m = 1", linewidth=2)
    #plt.plot(n_ch, m256_snr, '-o', label="M256, m = 1", linewidth=2)
    ax2.set_xlabel("n")
    ax2.set_ylabel("SNR")
    ax2.set_xlim([0, n_ch[-1]])
    ax2.grid()
    ax2.legend()
    plt.savefig('/home/vereese/thesis_pics/msk_snr.eps', bbox_inches='tight')

    phi = np.arange(0, 1, 1/len(I.profile))
    fig3, ax3 = plt.subplots()
    ax3.plot(phi, I.norm_profile + 0.5, label="none", linewidth=2)
    #ax3.plot(phi, sk_4sig["64"].norm_profile+0.5, label="SK, M = 64, PFA: 4$\sigma$")
    #ax3.plot(phi, sk_4sig["128"].norm_profile+0.6, label="SK, M = 128, PFA: 4$\sigma$")
    #ax3.plot(phi, sk_4sig["256"].norm_profile+0.7, label="SK, M = 256, PFA: 4$\sigma$")
    ax3.plot(phi, sk_4sig["512"].norm_profile + 0.4, label="SK, PFA: 4$\sigma$", linewidth=2)
    #ax3.plot(phi, sk_4sig["1024"].norm_profile+0.4, label="SK, M = 1024, PFA: 4$\sigma$")
    #ax3.plot(phi, sk_4sig["2048"].norm_profile+1, label="SK, M = 2048, PFA: 4$\sigma$")
    ax3.plot(phi, pt.norm_profile + 0.3, label=">= 4$\sigma$", linewidth=2)
    #ax3.plot(phi, sk_pfa4sigskmaxlim["1024"].norm_profile + 0.3, label="SK, PFA: 4$\sigma$ SK max")
    #ax3.plot(phi, sk_low_4sig["1024"].norm_profile + 0.2, label="SK, PFA: low 4$\sigma$")
    ax3.plot(phi, sk_l1siguskmax["512"].norm_profile + 0.2, label="SK, PFA: 1$\sigma$, $SK_{max}$", linewidth=2)
    ax3.plot(phi, med.norm_profile + 0.1, label="median", linewidth=2)
    #ax3.plot(phi, sk_low_4sig_median["1024"].norm_profile, label="SK, PFA: low 4$\sigma$ median")
    ax3.set_ylabel("normalized pulsar intensity profile")
    ax3.set_xlabel("pulsar phase")
    ax3.set_xlim([0,1])
    ax3.grid()
    ax3.legend()
    plt.savefig('/home/vereese/thesis_pics/profile.eps', bbox_inches='tight')

    fig4, ax4 = plt.subplots()
    ax4.plot(frequencies, sk_4sig["512"].rfi_freq, label="SK, PFA: 4$\sigma$", linewidth=2)
    ax4.plot(frequencies, sk_l1siguskmax["512"].rfi_freq, label="SK, PFA: 1$\sigma$, $SK_{max}$", linewidth=2)
    ax4.plot(frequencies, med.rfi_freq, label="median", linewidth=2)
    ax4.plot(frequencies, pt.rfi_freq, label=">= 4$\sigma$", linewidth=2)
    ax4.set_ylabel("% RFI flagged")
    ax4.set_xlabel("frequency [MHz]")
    ax4.legend()
    ax4.set_xlim([frequencies[0], frequencies[-1]])
    ax4.set_ylim([0, 15])
    plt.axvspan(frequencies[0], frequencies[50], color='blue', alpha=0.5)
    plt.axvspan(frequencies[-50], frequencies[-1], color='blue', alpha=0.5)
    plt.axvspan(frequencies[95], frequencies[126], color='blue', alpha=0.5)
    plt.savefig('/home/vereese/thesis_pics/rfi_freq.eps', bbox_inches='tight')

    fig5, ax5 = plt.subplots()
    ax5.plot(phi, sk_4sig["512"].rfi_pulse, label="SK, PFA: 4$\sigma$", linewidth=2)
    ax5.plot(phi, sk_l1siguskmax["512"].rfi_pulse, label="SK, PFA: 1$\sigma$, $SK_{max}$", linewidth=2)
    ax5.plot(phi, med.rfi_pulse, label="median", linewidth=2)
    ax5.plot(phi, pt.rfi_pulse, label=">= 4$\sigma$", linewidth=2)
    ax5.set_ylabel("% RFI flagged")
    ax5.set_xlabel("pulse phase")
    ax5.legend()
    ax5.set_xlim([0, 1])
    plt.savefig('/home/vereese/thesis_pics/rfi_pulse.eps', bbox_inches='tight')
    plt.show()

"""a = I.I.sum(axis=1)
fig5, ax5 = plt.subplots()
ax5.plot(a - sk_4sig["2048"].I.sum(axis=1), label="sk pfa 4 sig")
ax5.plot(a - sk_low_4sig["2048"].I.sum(axis=1), label="sk low pfa 4 sig")
#ax5.plot(I.I.sum(axis=1), label="none")
ax5.plot(a - med.I.sum(axis=1), label="median")
ax5.plot(a - masked.I.sum(axis=1), label="masked")
plt.legend()
plt.show()
fi6, ax6 = plt.subplots()
ax6.plot(incoherent_dedisperse(sk_4sig["2048"].sf,"2210").mean(axis=0), label= "4 sig")
ax6.plot(incoherent_dedisperse(sk_nz4sig.sf, "2210").mean(axis=0), label="nz 4 sig")
ax6.plot(incoherent_dedisperse(sk_z4sig.sf, "2210").mean(axis=0), label="z 4 sig")
ax6.plot(incoherent_dedisperse(sk_zskmax.sf, "2210").mean(axis=0), label="z skmax")
ax6.plot(incoherent_dedisperse(sk_4sig["1024"].sf, "2210").mean(axis=0), label="sk 4 sig")
plt.legend()
plt.show()"""

#plt.plot(sk_pfa2["1024"].rfi, label="sk pfa 2 %")
#plt.semilogy(sk_low_pfa4sig["64"].rfi, label="sk low 4sig M64")
#plt.semilogy(sk_low_pfa4sig["128"].rfi, label="sk low 4sig M128")
#plt.semilogy(sk_low_pfa4sig["256"].rfi, label="sk low 4sig M256")
#ax4.semilogy(sk_low_pfa4sig["4096"].rfi, label="sk low 4sig M4096")
#plt.plot(sk_low_pfa4sig["256"].rfi, label="sk low 4sig M265")
#plt.plot(sk["1024"].rfi, label="sk 3sig ")
#plt.plot(vmsk_4siglow_M64m1nx["8"].rfi, label="msk M64m1n8")
#plt.semilogy(msk_4siglow_M256m1nx["2"].rfi, label="msk M256m1n2")
#plt.semilogy(vmsk_4siglow_M256m1n2.rfi, label="vmsk M256m1n2")
#plt.plot(sk_pfa4sigsklim["1024"].rfi, label="sk 4sig sklim")
#plt.plot(sk_pfa4sig4sigsklim["1024"].rfi, label="sk 4sig 4sig sklim")
#plt.plot(sk_4sig["64"].rfi, label="sk 4sig")
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
