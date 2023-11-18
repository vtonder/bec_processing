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
This script investigates the effect of fluctuating gain in noise across the band on the SK estimator
The fluctuating gain results in the noise distribution to becomes a Gaussian Mixture Model (GMM)   
'''

def add_gain(p, perc, frac):
    # Need to make a copy because python is pass by reference not value
    p2 = np.copy(p)
    p2[:, :int(M*frac)] *= (100 + perc)/100  # byvoorbeeld -> dit lig drywing met perc% die frac van tyd

    return p2

parser = argparse.ArgumentParser()
parser.add_argument("-M", dest="M", help="Number of spectra to accumulate in SK calculation", default=2048)
parser.add_argument("-n", dest="n", help="Number of SKs to calculate", default=100000)
parser.add_argument("-f", dest="frac", help="Number of SKs to calculate", default=0.5)
parser.add_argument("-l", dest="low", help="Key for lower threshold to use. Keys are defined constants file. Only 0 (3 sigma) and 7 (4 sigma) now supported.", default = 7)
parser.add_argument("-u", dest="up", help="Key for upper threshold to use. Keys are defined constants file. Only 0 (3 sigma), 7 (4 sigma), 8 (sk max) now supported.", default = 7)
parser.add_argument("-p", dest="plot", help="Plot or save the data", default=False)
args = parser.parse_args()

M = int(args.M)
num_sk = int(args.n)
frac = float(args.frac)
highest_gain = 8
step_size = 0.5
N = num_sk * M

low, low_prefix = get_low_limit(int(args.low), M)
up, up_prefix = get_up_limit(int(args.up), M)

t1 = time.time()
sk_values = np.arange(0, 2, 0.01)
theoretical_sk_pdf = sk_pdf(M, sk_values)

# Ludwig code
x = np.random.randn(num_sk, M) + 1j * np.random.randn(num_sk, M)
p1 = np.abs(x) ** 2
#wgn_re = np.random.normal(mean, std, size=N)
#wgn_im = np.random.normal(mean, std, size=N)
#x1 = (wgn_re + 1j*wgn_im) * np.asarray([1, 0] * int(N/2))

perc_flagged = []
perc_gain = np.arange(0, highest_gain, step_size)
sk_gain = []

for g in perc_gain:
    p2 = add_gain(p1, g, frac)

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

if args.plot:
    fig, ax = plt.subplots()
    ax.plot(perc_gain, perc_flagged, 'o')
    ax.set_ylabel("% RFI flagged")
    ax.set_xlabel("% Gain added")
    ax.set_title("PFA = " + low_prefix + up_prefix + " , M = " + str(M))

    ncol = 2
    nrow = int(len(perc_gain) / ncol)
    fig1, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True)
    axs = axs.flatten()

    for i in np.arange(nrow * ncol):
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
    np.save("gain_" + str(args.frac) + "_rfi_"+ low_prefix + up_prefix + "_M" + str(M) + "_d" + str(N), perc)
