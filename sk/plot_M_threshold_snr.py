import numpy as np
from pulsar_snr import PI
from matplotlib import pyplot as plt

t = ["1sig", "2sig", "2_5sig", "3sig", "4sig", "skmin"] # t : threshold
M = ["256", "2048"]

lt_snr = np.zeros([len(M), len(t)])
ut_snr = np.zeros([len(M), len(t)])

for i in np.arange(len(M)):
    # assume lower and upper thresholds have the same length
    for j in np.arange(len(t)):
        # set initialise to False and call compute - this way no mask is applied
        # TODO: but mask also affects snr for different M and threshold - so how does one then determine the best mask?
        intensity_l = PI("../", "sk_intensity_z_l" + t[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy",
                       "sk_num_nz_z_l" + t[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy", initialise = False)
        intensity_l.compute()
        intensity_u = PI("../", "sk_intensity_z_u" + t[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy",
                       "sk_num_nz_z_u" + t[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy", initialise = False)
        intensity_u.compute()
        lt_snr[i,j] = intensity_l.snr
        ut_snr[i,j] = intensity_u.snr

fig, ax = plt.subplots()
ax.imshow(lt_snr)
ax.title("Lower threshold")

fig1, ax1 = plt.subplots()
ax1.imshow(ut_snr)
ax1.title("Upper threshold")

plt.show()