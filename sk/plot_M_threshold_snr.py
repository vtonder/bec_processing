import numpy as np
from pulsar_snr import PI
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", dest = "dir", help="Directory where data is located. default: /net/com08/data6/vereese/phd_data/sk_analysis/2210/", default="/net/com08/data6/vereese/phd_data/sk_analysis/2210/")
args = parser.parse_args()


lt = ["0sig", "0_5sig","1sig", "2sig", "2_5sig", "3sig", "4sig", "skmin"] # lt : lower threshold
ut = ["0sig", "0_5sig","1sig", "2sig", "2_5sig", "3sig", "4sig", "skmax"] # ut : upper threshold

M = ["64", "128", "256", "512", "1024", "2048", "4096", "8192"]

lt_snr = np.zeros([len(M), len(lt)])
ut_snr = np.zeros([len(M), len(ut)])

for i in np.arange(len(M)):
    # assume lower and upper thresholds have the same length
    print("\nMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM: ", M[i])

    for j in np.arange(len(lt)):
        # set initialise to False and call compute - this way no mask is applied
        # TODO: but mask also affects snr for different M and threshold - so how does one then determine the best mask?
        print("lt: ", lt[j])
        print("")

        intensity_l = PI(args.dir, "sk_intensity_z_l" + lt[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy", "sk_num_nz_z_l" + lt[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy")
        #intensity_l = PI(args.dir, "sk_intensity_z_l" + lt[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy", "sk_num_nz_z_l" + lt[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy", initialise = False)
        #intensity_l.compute()

        print("ut: ", ut[j])
        print("")

        intensity_u = PI(args.dir, "sk_intensity_z_u" + ut[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy", "sk_num_nz_z_u" + ut[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy")
        #intensity_u = PI(args.dir, "sk_intensity_z_u" + ut[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy", "sk_num_nz_z_u" + ut[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy", initialise = False)
        #intensity_u.compute()

        lt_snr[i,j] = intensity_l.snr
        ut_snr[i,j] = intensity_u.snr

lt_labels = ("$0\sigma$", "$0.5\sigma$", "$1\sigma$", "$2\sigma$", "$2.5\sigma$", "$3\sigma$", "$4\sigma$", "$SK_{min}$")
x_range = np.arange(len(lt_labels))
y_range = np.arange(len(M))
fig, ax = plt.subplots()
snr_im_lt = ax.imshow(lt_snr, origin="lower")#, aspect="auto", extent=[1,6,128,8192])
cbar = fig.colorbar(snr_im_lt, ax=ax)
cbar.minorticks_on()
ax.set_ylabel("M")
ax.set_xlabel("threshold")
ax.set_title("Lower threshold")
ax.set_xticks(x_range, lt_labels)
ax.set_yticks(y_range, M)

ut_labels = ("$0\sigma$", "$0.5\sigma$", "$1\sigma$", "$2\sigma$", "$2.5\sigma$", "$3\sigma$", "$4\sigma$", "$SK_{max}$")
fig1, ax1 = plt.subplots()
snr_im_ut = ax1.imshow(ut_snr, origin="lower")#, aspect="auto", extent=[1,6,128,8192])
cbar1 = fig.colorbar(snr_im_ut, ax=ax1)
cbar1.minorticks_on()
ax1.set_title("Upper threshold")
ax1.set_xticks(x_range, ut_labels)
ax1.set_yticks(y_range, M)
ax1.set_ylabel("M")
ax1.set_xlabel("threshold")
plt.show()
