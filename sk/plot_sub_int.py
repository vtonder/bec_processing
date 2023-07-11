import numpy as np
from matplotlib import pyplot as plt

font = {'family': 'STIXGeneral',
        'size': 42}
plt.rc('font', **font)

s = np.load("sub_int_intensity_2210.npy")
s1 = np.load("sub_int_SK_intensity_M1024_2210.npy")

# identified ch 280 as a intermittent RFI ch
# wrt observation time:
# 157*9*32*J0437_samples_T*time_resolution*10**-6 = 260 
# 260 / 60 = 4.3

plt.figure(1, figsize=[22,16])
plt.imshow(s1[:,280,:],origin="lower",aspect="auto", extent=[0, 1, 0, 4.3])
plt.xlabel("pulsar phase")
plt.ylabel("observation time [min]")
plt.savefig('/home/vereese/sub_int_sk.eps', bbox_inches='tight')

plt.figure(1, figsize=[22,16])
plt.imshow(s[:,280,:],origin="lower",aspect="auto", extent=[0, 1, 0, 4.3])
plt.xlabel("pulsar phase")
plt.ylabel("observation time [min]")
plt.savefig('/home/vereese/sub_int.eps', bbox_inches='tight')
