from sk.pulsar_snr import PI
from matplotlib import pyplot as plt
#import sys
#sys.path.append("../")
#sys.path.append("../pulsar_processing")
from pulsar_processing.pulsar_functions import incoherent_dedisperse
import numpy as np


direc = "/home/vereese/data/phd_data/sk_analysis/2210/"
M = ["256", "512", "1024", "2048", "4096"]
data = {}
for i, m in enumerate(M):
    isk = PI(direc, "sk_intensity_z_l1siguskmax_M" + m + "_m1_n1_2210_p45216.npy", "sk_num_nz_z_l1siguskmax_M" + m + "_m1_n1_2210_p45216.npy", "sk_summed_flags_z_l1siguskmax_M" + m + "_m1_n1_2210_p45216.npy")
    data.update({m: isk})
    dsf = incoherent_dedisperse(np.copy(isk.sf), "2210")
    mx = np.max(dsf)
    mn = np.min(dsf)

    plt.figure(i)
    plt.imshow(dsf, origin = "lower", aspect = "auto", vmin = mn, vmax = mx / 8)
    plt.title("M = " + m)

plt.show()
