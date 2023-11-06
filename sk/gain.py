import numpy as np
import sys
sys.path.append("../")
from kurtosis import spectral_kurtosis_cm_perio
from matplotlib import pyplot as plt
from common_sk import rfi_mitigation, get_low_limit, get_up_limit
from investigate_sk import sk_pdf
import time
import argparse
'''
This script investigates the effect of fluctuating gain in noise on sk
Becomes a Gaussian Mixture Model (GMM) because 
Effect of mixture Gaussian noise on SK  
'''

def add_gain(p, perc):
    p2 = p
    p2[:, :int(M / 2)] *= (100+perc)/100  # byvoorbeeld -> dit lig drywing met 0.5% die helfte van tyd
    return p2

parser = argparse.ArgumentParser()
parser.add_argument("-M", dest="M", help="Number of spectra to accumulate in SK calculation", default=2048)
parser.add_argument("-n", dest="n", help="Number of SKs to calculate", default=100000)
parser.add_argument("-l", dest="low", help="Key for lower threshold to use. Keys are defined constants file. Only 0 (3 sigma) and 7 (4 sigma) now supported.", default=7)
parser.add_argument("-u", dest="up", help="Key for upper threshold to use. Keys are defined constants file. Only 0 (3 sigma), 7 (4 sigma), 8 (sk max) now supported.", default=7)
parser.add_argument("-p", dest="plot", help="Plot or save the data", default=False)
args = parser.parse_args()

M = args.M
num_sk = args.n
highest_gain = 5
step_size = 0.5
N = num_sk * M

low, low_prefix = get_low_limit(args.low, M)
up, up_prefix = get_up_limit(args.up, M)

t1 = time.time()
sk_values = np.arange(0, 2, 0.01)
theoretical_sk_pdf = sk_pdf(M, sk_values)
#pdf_M512 = np.load("/home/vereese/git/phd_data/sk_analysis/2210/pdf_M512.npy")

# Ludwig code
x = np.random.randn(num_sk, M) + 1j * np.random.randn(num_sk, M)
#p1 = x * x.conj()
p1 = np.abs(x) ** 2
#wgn_re = np.random.normal(mean, std, size=N)
#wgn_im = np.random.normal(mean, std, size=N)
#x1 = (wgn_re + 1j*wgn_im) * np.asarray([1, 0] * int(N/2))

perc_flagged = []
perc_gain = np.arange(0, highest_gain, step_size)
sk_gain = []

for g in perc_gain:
    p2 = add_gain(p1, g)

    # wgn2_re = np.random.normal(mean, std2, size=N)
    # wgn2_im = np.random.normal(mean, std2, size=N)
    # x2 = (wgn2_re + 1j*wgn2_im) * np.asarray([0, 1] * int(N/2))
    # x = x1 + x2

    sk = spectral_kurtosis_cm_perio(p2, M)
    flags = np.zeros(num_sk)
    for i, val in enumerate(sk):
        if val < low or val > up:
            flags[i] = 1

    sk_gain.append(sk)
    pf = 100*np.sum(flags) / num_sk
    print("gain: ", g, "\n% flagged: ", pf)
    perc_flagged.append(pf)

print("Done processing: ", time.time()-t1)
#data, sk_flags, sf = rfi_mitigation(data, M, data_window_len, std, check_thres, sk_flags, summed_flags, ndp, chunk_start, first_non_zero_idx):
if args.plot:
    fig, ax = plt.subplots()
    ax.plot(perc_gain, perc_flagged, 'o')
    ax.set_ylabel("% RFI flagged")
    ax.set_xlabel("% Gain added")
    ax.set_title("PFA = 0.27% , M = " + str(M))

    ncol = 2
    nrow = int(len(perc_gain)/ncol)
    fig1, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True)
    axs = axs.flatten()

    for i in np.arange(nrow*ncol):
        axs[i].hist(sk_gain[i], 100, density=True, log=True, stacked=True)
        axs[i].set_title(str(perc_gain[i]) + " % gain added")
        axs[i].plot(sk_values, theoretical_sk_pdf)
        axs[i].set_ylim([10**-2, 10**1])
        #axs[i].vlines(x=low) #,xmin=M[0],xmax=M[-1], colors="red", linestyle="--", label="none")
        #ax1.set_ylabel("SK")
        #ax1.set_xlabel("% Ga")
    fig1.supxlabel("SK values")
    fig1.supylabel("PDF")
    plt.show()
else:
    perc = np.asarray([perc_gain, perc_flagged])
    print("% gain       : ", perc_gain)
    print("% RFI flagged: ", perc_flagged)
    np.save("gain_rfi_"+ low_prefix + up_prefix + "_M" + str(M) + "_d" + str(N), perc)