import numpy as np
from matplotlib import pyplot as plt

DIR1 = "/home/vereese/git/phd_data/"
DIR2 = "/home/vereese/git/phd_data/sk_analysis/1569/"

#vela = np.load(DIR1+"intensity_1569.npy")
#vela_sk = np.load("itegrated_sk_low_pfa_1_intensity_M2048_1569.npy")
vela_sk = np.load(DIR2+"itegrated_sk_intensity_diff_M16384_1569.npy")
sf = np.load(DIR2+"sk_summed_flags_diff_M16384_1569.npy")

#profile_vela = vela.sum(axis=0)
profile_vela_sk = vela_sk.sum(axis=0)

plt.figure(0)
#plt.plot(profile_vela, label="None")
plt.plot(profile_vela_sk, label="SK")
plt.legend()

plt.figure(1)
plt.imshow(sf, origin="lower", aspect="auto")
plt.show()
