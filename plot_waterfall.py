import numpy as np
from matplotlib import pyplot as plt
import argparse

textwidth = 9.6 # 128.0 / 25.4 #
textheight = 7 # 96.0 / 25.4 # 7
plt.rc('font', size=12, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=12, labelsize=12)
plt.rc(('xtick', 'ytick'), labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

data = np.load("/home/vereese/data/phd_data/waterfall_1569_0x_1638400.npy")

plt.figure()
plt.imshow(data, origin="lower", aspect="auto", extent=[0, 1, 0, 4.3])
plt.xlabel("observation time [min]")
plt.ylabel("frequency [MHz]")
plt.savefig('/home/vereese/waterfall2.png', bbox_inches='tight')
#plt.show()
