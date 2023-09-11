import numpy as np
from matplotlib import pyplot as plt

DIR1 = "/home/vereese/git/phd_data/"
DIR2 = "/home/vereese/git/phd_data/sk_analysis/1569/"

vela = np.load(DIR1+"intensity_1569.npy")
vela_sk = np.load(DIR2+"itegrated_sk_low_pfa_1_intensity_M2048_1569.npy")

profile_vela = vela.sum(axis=0)
profile_vela_sk = vela_sk.sum(axis=0)

plt.figure()
plt.plot(vela, label="None")
plt.plot(vela_sk, label="SK")
plt.legend()
plt.show()