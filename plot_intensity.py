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

#data = np.load("/home/vereese/git/phd_data/intensity_" + args.tag + ".npy")
DIR = "/home/vereese/data/phd_data/sk_analysis/1234/lower/"
d1 = np.load(DIR + "SK_flags512_1234_0x.npy")
d2 = np.load(DIR + "SK_flags1024_1234_0x.npy")
d3 = np.load(DIR + "SK_flags2048_1234_0x.npy")
d4 = np.load(DIR + "SK_flags10240_1234_0x.npy")
data = np.load("intensity_xpol_1234.npy")

for i in np.arange(1024):
    data[i,:] = data[i,:] - np.mean(data[i,:])
#profile = data.sum(axis=0)
#mp = list(profile).index(max(list(profile)))
#print("max point index: ", mp)
#num_2_roll = int(len(profile)/2 - mp)
#maxi = np.max(data[300:600,:])
#mini = np.min(data[300:600,:])

profile = data.sum(axis=0)
mp = list(profile).index(max(list(profile)))
print("max point index: ", mp)
num_2_roll = int(len(profile)/2 - mp)

plt.figure(0)
plt.imshow(np.roll(data,num_2_roll), origin="lower", aspect="auto", extent=[0, 1, 856, 1712])#, vmin=mini, vmax=maxi)
plt.xlabel("pulse phase")
plt.ylabel("frequency [MHz]")
plt.savefig('/home/vereese/far', bbox_inches='tight')

'''plt.figure(0)
plt.imshow(d1, origin="lower", aspect="auto", extent=[0, 10, 856, 1712])
plt.xlabel("observation time [s]")
plt.ylabel("frequency [MHz]")
plt.savefig('/home/vereese/sk_flags_512', bbox_inches='tight')

plt.figure(1)
plt.imshow(d2, origin="lower", aspect="auto", extent=[0, 10, 856, 1712])
plt.xlabel("observation time [s]")
plt.ylabel("frequency [MHz]")
plt.savefig('/home/vereese/sk_flags_1024', bbox_inches='tight')

plt.figure(3)
plt.imshow(d2, origin="lower", aspect="auto", extent=[0, 10, 856, 1712])
plt.xlabel("observation time [s]")
plt.ylabel("frequency [MHz]")
plt.savefig('/home/vereese/sk_flags_2048', bbox_inches='tight')

plt.figure(2)
plt.imshow(d3, origin="lower", aspect="auto", extent=[0, 10, 856, 1712])
plt.xlabel("observation time [s]")
plt.ylabel("frequency [MHz]")
plt.savefig('/home/vereese/sk_flags_10240', bbox_inches='tight')'''
