import numpy as np
from matplotlib import pyplot as plt
import argparse
from common import mean_compensation
from constants import thesis_font, a4_textwidth, a4_textheight

# Setup fonts and sizes for publication, based on page dimensions in inches
textwidth =  a4_textwidth
textheight = a4_textheight
font_size = thesis_font
# groups are like plt.figure plt.legend etc
plt.rc('font', size=font_size, family='serif')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=font_size, labelsize=font_size)
plt.rc(('xtick', 'ytick'), labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('lines', markersize=5)
# The following should only be used for beamer
# plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
figheight = 0.65 * textwidth
plt.rc('mathtext', fontset='cm')
# to get this working needed to do: sudo apt install cm-super
plt.rc("text", usetex = True)
plt.rc("figure", figsize = (textwidth, figheight))

parser = argparse.ArgumentParser()
parser.add_argument("tag", help="observation tag to plot intensity")
parser.add_argument("-d", dest="dir", help="directory where intensity file is located", default="/home/vereese/git/phd_data/")
args = parser.parse_args()

#data = np.load("/home/vereese/git/phd_data/intensity_" + args.tag + ".npy")
DIR_OUT = "/home/vereese/thesis_pics/"

"""
# following 14 lines of code is for plotting far.pdf, an image in lit study of thesis
data = np.load("intensity_xpol_1234.npy")
for i in np.arange(1024):
    data[i,:] = data[i,:] - np.mean(data[i,:])
profile = data.sum(axis=0)
mp = list(profile).index(max(list(profile)))
print("max point index: ", mp)
num_2_roll = int(len(profile)/2 - mp)
maxi = np.max(data)/2
mini = np.min(data)
plt.figure(0)
plt.imshow(np.roll(data,num_2_roll), origin="lower", aspect="auto", extent=[0, 1, 856, 1712], vmin=mini, vmax=maxi)
plt.xlabel("pulse phase")
plt.ylabel("frequency [MHz]")
plt.savefig(DIR_OUT + "far.pdf", bbox_inches='tight')

# following 29 lines of code is for plotting SK flags produced during vela analysis for URSI 2023 
DIR = "/home/vereese/data/phd_data/sk_analysis/1234/lower/"
d1 = np.load(DIR + "SK_flags512_1234_0x.npy")
d2 = np.load(DIR + "SK_flags1024_1234_0x.npy")
d3 = np.load(DIR + "SK_flags2048_1234_0x.npy")
d4 = np.load(DIR + "SK_flags10240_1234_0x.npy")

plt.figure(0)
plt.imshow(d1, origin="lower", aspect="auto", extent=[0, 10, 856, 1712])
plt.xlabel("observation time [s]")
plt.ylabel("frequency [MHz]")
plt.savefig(DIR_OUT + 'sk_flags_512.pdf', bbox_inches='tight')

plt.figure(1)
plt.imshow(d2, origin="lower", aspect="auto", extent=[0, 10, 856, 1712])
plt.xlabel("observation time [s]")
plt.ylabel("frequency [MHz]")
plt.savefig(DIR_OUT + 'sk_flags_1024.pdf', bbox_inches='tight')

plt.figure(3)
plt.imshow(d2, origin="lower", aspect="auto", extent=[0, 10, 856, 1712])
plt.xlabel("observation time [s]")
plt.ylabel("frequency [MHz]")
plt.savefig(DIR_OUT + 'sk_flags_2048.pdf', bbox_inches='tight')

plt.figure(2)
plt.imshow(d3, origin="lower", aspect="auto", extent=[0, 10, 856, 1712])
plt.xlabel("observation time [s]")
plt.ylabel("frequency [MHz]")
plt.savefig(DIR_OUT + 'sk_flags_10240.pdf', bbox_inches='tight')
"""

#data_sk = np.load("/home/vereese/git/phd_data/sk_analysis/2210/itegrated_sk_intensity_M4096_" + args.tag + ".npy")
#data_var = np.load("/home/vereese/git/phd_data/sk_analysis/2210/var_threshold_intensity_M4096_" + args.tag + ".npy")
#data = np.load("/home/vereese/git/phd_data/intensity_1569.npy")
#data = np.load("/home/vereese/git/phd_data/pulsar/sub_int_vela_11.1946499395.npy") #sub_integration_true_period_1569.npy")
#data_sk = np.load("/home/vereese/git/phd_data/sk_analysis/2210/itegrated_sk_intensity_M2048_2210.npy")
#data_b = np.load("/home/vereese/git/phd_data/sk_analysis/2210/itegrated_sk_bright_intensity_M2048_2210.npy")
#data_b1 = np.load("/home/vereese/git/phd_data/sk_analysis/2210/itegrated_sk_bright_intensity_2210.npy")
#data = np.load("/home/vereese/git/phd_data/sk_analysis/2210/intensity_M8192_2210.npy")
#data1 = np.load("/home/vereese/git/phd_data/sk_analysis/2210/itegrated_sk_intensity_M8192_2210.npy")
#vela_sk = np.load("/home/vereese/git/phd_data/sk_analysis/1569/itegrated_sk_low_pfa_1_intensity_M16384_1569.npy")
#vela = np.load("/home/vereese/git/phd_data/sk_analysis/1569/itegrated_sk_low_pfa_1_intensity_M16384_1569.npy")
#vela = np.load("/home/vereese/git/phd_data/intensity_1569.npy")
#for i in np.arange(1024):
#    data[i,:] = data[i,:] - np.mean(data[i,:])
#profile = vela.sum(axis=0)

#data = np.load(args.dir+"intensity_"+args.tag+".npy")
#data = mean_compensation(data)
#profile = data.sum(axis=0)
#profile_sk = data_sk.sum(axis=0)
#profile_b = data_b.sum(axis=0)
#profile_b1 = data_b1.sum(axis=0)
#mp = list(profile).index(max(list(profile)))
#print("max point index: ", mp)
#num_2_roll = int(len(profile)/2 - mp)
#profile1 = data1[0:200,:].sum(axis=0)
#len = profile_b1.shape[0]
#c = np.correlate(profile_b, profile_b1, "same")
"""range = np.arange(-len/2,len/2)
plt.figure(0)
plt.plot(range,c)
plt.title("correlation")
plt.grid()
plt.xlim([range[0], range[-1]])
mini = min(data.flatten())
maxi = max(data.flatten())/4
plt.figure(0)
#plt.imshow(np.roll(data, num_2_roll), origin="lower", aspect="auto", extent=[0, 1, 856, 1712])#,vmax=maxi,vmin=mini)
plt.imshow(data, origin="lower", aspect="auto", extent=[0, 1, 856, 1712])#,vmax=maxi,vmin=mini)
#plt.plot(data_b.sum(axis=0), label="bright sk")#, extent=[0, 1, 856, 1712])
#plt.plot(data_b1.sum(axis=0), label="bright none")#, extent=[0, 1, 856, 1712])
#plt.plot(data_sk.sum(axis=0), label="sk")#, extent=[0, 1, 856, 1712])
#plt.plot(data.sum(axis=0), label="none")#, extent=[0, 1, 856, 1712])
#plt.plot(data_b[130:345,:].sum(axis=0), label="None")
#plt.plot(profile_sk, label="SK")
#plt.plot(profile_var, label="sigma")
#plt.xlim([0, len])
plt.xlabel("pulse phase")
plt.ylabel("frequency [MHz]")
#plt.ylabel("time [min]")
#plt.savefig('/home/vereese/Documents/PhD/URSI2023/conference_paper/velask.png', bbox_inches='tight')
plt.show()"""
