import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
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

# identified ch 280 as an intermittent RFI ch
# wrt observation time:
# 157*9*32*J0437_samples_T*time_resolution*10**-6 = 260 
# 260 / 60 = 4.3
num_sub_ints = s1.shape[0]
num_2_roll = int(s1.shape[2]/2 - list(s1[:,280,:].sum(axis=0)).index(max(list(s1[:,280,:].sum(axis=0)))))
print("roll by: ", num_2_roll)

plt.figure(0) #, figsize=[22,16])
plt.imshow(np.roll(s1[:,280,:], num_2_roll), origin="lower", aspect="auto", extent=[0, 1, 0, 4.3])
func = lambda x, pos: "" if np.isclose(x,0) else x
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(func))
plt.xlabel("pulsar phase")
plt.ylabel("observation time [min]")
plt.savefig('/home/vereese/sub_int_sk.eps', bbox_inches='tight')

plt.figure(1) #, figsize=[22,16])
plt.imshow(np.roll(s[:,280,:], num_2_roll), origin="lower", aspect="auto", extent=[0, 1, 0, 4.3])
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(func))
plt.xlabel("pulsar phase")
plt.ylabel("observation time [min]")
plt.savefig('/home/vereese/sub_int.eps', bbox_inches='tight')
