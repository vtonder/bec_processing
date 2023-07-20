import numpy as np
from matplotlib import pyplot as plt

#font = {'family': 'STIXGeneral',
#        'size': 42}
#plt.rc('font', **font)

textwidth = 9.6 #128.0 / 25.4
textheight = 7 #96.0 / 25.4
plt.rc('font', size=22, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=22, labelsize=22)
plt.rc(('xtick', 'ytick'), labelsize=22)
plt.rc('legend', fontsize=22)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

s = np.load("sub_int_intensity_2210.npy")
s1 = np.load("sub_int_SK_intensity_M1024_2210.npy")

# identified ch 280 as a intermittent RFI ch
# wrt observation time:
# 157*9*32*J0437_samples_T*time_resolution*10**-6 = 260 
# 260 / 60 = 4.3

plt.figure(0) # , figsize=[22,16])
plt.imshow(s1[:,280,:],origin="lower",aspect="auto", extent=[0, 1, 0, 4.3])
plt.xlabel("pulsar phase")
plt.ylabel("observation time [min]")
plt.savefig('/home/vereese/sub_int_sk.eps', bbox_inches='tight')

plt.figure(1) #, figsize=[22,16])
plt.imshow(s[:,280,:],origin="lower",aspect="auto", extent=[0, 1, 0, 4.3])
plt.xlabel("pulsar phase")
plt.ylabel("observation time [min]")
plt.savefig('/home/vereese/sub_int.eps', bbox_inches='tight')
