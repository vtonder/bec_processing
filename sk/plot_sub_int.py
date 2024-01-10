import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import sys
sys.path.append('../')
from constants import a4_textwidth, a4_textheight, thesis_font

# Setup fonts and sizes for publication, based on page dimensions in inches
textwidth = a4_textwidth
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

dir = "/home/vereese/data/phd_data/sk_analysis/2210/"

s_none = np.load(dir + "sub_int_intensity_2210.npy")
s_sk256 = np.load(dir + "sub_int_intensity_M16384_2210.npy")
s_pt = np.load(dir + "sub_int_intensity_sigma2210.npy")

# identified ch 280 as an intermittent RFI ch
# wrt observation time:
# 45216 (total # pulses) * J0437_samples_T*time_resolution*10**-6 = 260 
# 260 / 60 = 4.3
num_sub_ints = s_sk256.shape[0]
num_2_roll = int(s_sk256.shape[2]/2 - list(s_sk256[:,280,:].sum(axis=0)).index(max(list(s_sk256[:,280,:].sum(axis=0)))))
print("roll by: ", num_2_roll)

plt.figure(0)
plt.imshow(np.roll(s_sk256[:,280,:], num_2_roll), origin="lower", aspect="auto", extent=[0, 1, 0, 4.3])
func = lambda x, pos: "" if np.isclose(x,0) else x
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(func))
plt.xlabel("pulsar phase")
plt.ylabel("observation time [min]")
plt.savefig('/home/vereese/thesis_pics/sub_int_sk_M256_l1siguskmax.pdf', bbox_inches='tight')

plt.figure(1)
plt.imshow(np.roll(s_none[:,280,:], num_2_roll), origin="lower", aspect="auto", extent=[0, 1, 0, 4.3])
func = lambda x, pos: "" if np.isclose(x,0) else x
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(func))
plt.xlabel("pulsar phase")
plt.ylabel("observation time [min]")
plt.savefig('/home/vereese/thesis_pics/sub_int_none.pdf', bbox_inches='tight')

plt.figure(2)
plt.imshow(np.roll(s_pt[:,280,:], num_2_roll), origin="lower", aspect="auto", extent=[0, 1, 0, 4.3])
func = lambda x, pos: "" if np.isclose(x,0) else x
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(func))
plt.xlabel("pulsar phase")
plt.ylabel("observation time [min]")
plt.savefig('/home/vereese/thesis_pics/sub_int_pt.pdf', bbox_inches='tight')
