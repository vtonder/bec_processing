import numpy as np
from matplotlib import pyplot as plt

def zero_mask(intensity):
    intensity[0:50, :] = np.zeros([50, 4812])
    intensity[95:125, :] = np.zeros([30, 4812])
    intensity[347:532, :] = np.zeros([185, 4812])
    intensity[794:900, :] = np.zeros([106, 4812])
    intensity[-50:, :] = np.zeros([50, 4812])

    return intensity
M = ["64", "128", "256", "512", "1024", "2048", "4096", "8192", "16384"]
M = ["4096"]
DIR = "/home/vereese/git/phd_data/sk_analysis/2210/"
I = np.load(DIR+"intensity_2210.npy")
sk_low_pfa4sig = zero_mask(np.load(DIR+"sk_intensity_low_pfa4sig_M2048_2210.npy"))
sk_pfa4sig = zero_mask(np.load(DIR+"sk_intensity_pfa4sig_M2048_2210.npy"))
med = np.load(DIR+"power_mit_median_smoothed_I4.npy")
power_sig4skmaxlim = {}
power_mit_sig4skmaxlim = {}
for m in M:
    print(m)
    power_sig4skmaxlim.update({m:np.load(DIR+"summed_power_low_sig4_M"+m+"_2210.npy")})
    power_mit_sig4skmaxlim.update({m:np.load(DIR+"summed_mit_power_low_sig4_M"+m+"_2210.npy")})

#print(np.any(power_mit_sig4skmaxlim["512"]/max(power_mit_sig4skmaxlim["512"]) - np.ones(1024)))
plt.figure()
#plt.plot(I.sum(axis=1), label="total power via pulsar intensity")
#plt.plot(power_sig4skmaxlim["512"], label="power")
plt.plot(power_sig4skmaxlim["4096"], label="power 4096")
plt.plot(power_mit_sig4skmaxlim["4096"], label="mit power 4096")
plt.axvspan(0,50)
plt.axvspan(95,125)
plt.axvspan(347,532)
plt.axvspan(794,900)
plt.axvspan(974,1024)
#plt.plot(power_mit_sig4skmaxlim["1024"], label="mit power 1024")
#plt.plot(power_mit_sig4skmaxlim["2048"], label="mit power 2048")
#plt.plot(sk_low_pfa4sig.sum(axis=1)[910:925], label="sk low 4 sig")
#plt.plot(sk_pfa4sig.sum(axis=1)[910:925], label="sk 4 sig")
plt.plot(med, label="med")
plt.legend()
plt.show()