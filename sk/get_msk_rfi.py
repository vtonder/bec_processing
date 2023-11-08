import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("SK", help="Type of SK. ie VMSK or MSK", default="msk")
parser.add_argument("M", help="Number of spectra to accumulate in SK calculation", default=512)
parser.add_argument("m", help="Number of time samples to add up in MSK", default=1)
parser.add_argument("n", help="Number of ch to add up in MSK", default=1)

args = parser.parse_args()
SK = str(args.SK)
M = str(args.M)
m = str(args.m)
n = str(args.n)

xdata = np.load(SK + "_xpol_flags_l4sigu4sig_M" + M + "_m" + m + "_n" + n + "_2210_p45216.npy")
ydata = np.load(SK + "_ypol_flags_l4sigu4sig_M" + M + "_m" + m + "_n" + n + "_2210_p45216.npy")

print("x shape: ", xdata.shape)
print("y shape: ", ydata.shape)

rfi = 100*(np.sum(np.where(xdata >= 1, True, False), axis=1) + np.sum(np.where(ydata >= 1, True, False), axis=1))/(xdata.shape[1] + ydata.shape[1])
rfi_x = 100 * (np.sum(np.where(xdata >= 1, True, False), axis=1)) / xdata.shape[1]
rfi_y = 100 * (np.sum(np.where(ydata >= 1, True, False), axis=1)) / ydata.shape[1]

np.save(SK + "_rfi_xpol_" + M + "_m" + m + "_n" + n, rfi_x)
np.save(SK + "_rfi_ypol_" + M + "_m" + m + "_n" + n, rfi_y)
np.save(SK + "_rfi_" + M + "_m" + m + "_n" + n, rfi)

