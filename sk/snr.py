import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from constants import time_resolution

font = {'family': 'STIXGeneral',
        'size': 42}
plt.rc('font', **font)

DIR = "/home/vereese/git/phd_data/sk_analysis/2210/"
file_names = ["intensity_2210.npy", "itegrated_sk_intensity_M512_2210.npy", "itegrated_sk_intensity_M1024_2210.npy",
              "itegrated_sk_intensity_M2048_2210.npy", "itegrated_sk_intensity_M4096_2210.npy",
              "itegrated_sk_intensity_M8192_2210.npy"]

pulse_start = 2900
pulse_stop = 3250
pulse_w = pulse_stop - pulse_start

power_spectrum = {}
profile = {}

for fn in file_names:
    ps = np.load(DIR+fn)
    p = ps.sum(axis=0)
    m = np.mean(p[0:pulse_w])
    s = np.std(p[0:pulse_w])
    power_spectrum.update({fn:ps})
    profile.update({fn:p})
    snr = 0
    for i in np.arange(pulse_w):
        snr = snr + (p[pulse_start+i] - m)
    snr = snr / (s*np.sqrt(pulse_w))
    print("Analysis for: ", fn)
    print("SNR         : ", snr)
    print("TOA un [us] : ", (pulse_w*time_resolution)/snr)

mp = list(profile["intensity_2210.npy"]).index(max(list(profile["intensity_2210.npy"])))
print("max point index: ", mp)
num_2_roll = int(len(profile["intensity_2210.npy"])/2 - mp)
print("roll by", num_2_roll)
labels = ["None", "M = 512", "M = 1024", "M = 2048", "M = 4096", "M = 8192"]
phi = np.arange(0,1,1/len(profile["intensity_2210.npy"]))
plt.figure(1, figsize=[22,16])
for i, p in enumerate(profile.values()):
    plt.plot(phi, (np.roll(p, num_2_roll))/max(p), label=labels[i])
plt.grid()
plt.legend()
plt.xlim([0,1])
plt.xlabel("pulsar phase")
plt.ylabel("pulsar profile")
#plt.savefig('/home/vereese/Documents/PhD/jai-2e/profiles.eps', bbox_inches='tight')
plt.show()