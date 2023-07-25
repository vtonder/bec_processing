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

parser = argparse.ArgumentParser()
parser.add_argument("tag", help="observation tag to plot intensity")
args = parser.parse_args()

data = np.load("/home/vereese/git/phd_data/intensity_" + args.tag + ".npy")
#for i in np.arange(1024):
#    data[i,:] = data[i,:] - np.mean(data[i,:])
#profile = data.sum(axis=0)
#mp = list(profile).index(max(list(profile)))
#print("max point index: ", mp)
#num_2_roll = int(len(profile)/2 - mp)


plt.figure()
plt.imshow(data, origin="lower", aspect="auto", extent=[0, 1, 856, 1712])
plt.xlabel("pulse phase")
plt.ylabel("frequency [MHz]")
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/p4511.eps', bbox_inches='tight')
plt.show()
