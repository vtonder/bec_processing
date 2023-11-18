import numpy as np
import time
import sys
sys.path.append('../')
from kurtosis import ms_spectral_kurtosis_cm
import argparse

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-M", dest = "M", help="Number of spectra to accumulate in (M)SK calculation", default=512)
parser.add_argument("-m", dest = "m", help="Number of time samples to add up in MSK", default=1)
parser.add_argument("-n", dest = "n", help="Number of ch to add up in MSK", default=1)
parser.add_argument("-f", dest = "file_prefix", help="prefix to the output files", default="sk")
parser.add_argument("-d", dest = "dp", help="how dropped packets were handled. g : replaced by Gaussian noise ; z : left as 0s ", default="z")

args = parser.parse_args()

tag = '_' + args.file[6:10] + '_'  # add last 4 digits of observation code onto the file_name
pol = args.file[-5:-3]  # polarisation 0x or 0y
M = int(args.M)
m = int(args.m)
n = int(args.n)

S1 = np.float32(np.load("S1_" + args.dp + "_M" + str(M) + tag + pol + ".npy"))
S2 = np.float32(np.load("S2_" + args.dp + "_M" + str(M) + tag + pol + ".npy"))

# MSK reduces to SK when m = n = 1 
sk = np.float32(ms_spectral_kurtosis_cm(S1, S2, M, m = m, n = n))

np.save(args.file_prefix + args.dp + '_M' + str(M) + "_m" + str(m) + "_n" + str(n) + tag + pol, sk)
print("procesing took: ", time.time() - t1)
