import h5py
import numpy as np
from matplotlib import pyplot as plt
from constants import time_resolution 

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

df = h5py.File('/net/com08/data6/vereese/1604641234_wide_tied_array_channelised_voltage_0x.h5', 'r')
dl=14336000-13516800

t = np.arange(dl)*time_resolution/10**3
plt.figure(0)
plt.plot(t, df['Data/bf_raw'][382,13516800:14336000,0], label='ch 382')
plt.plot(t, df['Data/bf_raw'][600,13516800:14336000,0], label='ch 600')
plt.grid()
plt.legend()
plt.xlim([t[0], t[-1]])
plt.xlabel("observation time [ms]")
plt.ylabel("real 8 bit data")
plt.savefig('/home/vereese/time_series.png', bbox_inches='tight')

plt.show()
