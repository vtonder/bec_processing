import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import sys
sys.path.append('../')
from constants import a4_textwidth, a4_textheight, thesis_font, jai_textwidth, jai_textheight, jai_font
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", dest = "dir", help="Directory where data is located. default: /net/com08/data6/vereese/phd_data/sk_analysis/2210/", default="/net/com08/data6/vereese/phd_data/sk_analysis/2210/")
args = parser.parse_args()

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
figheight = 0.65 * textwidth # used for thesis, the first 3 plots 
plt.rc('mathtext', fontset='cm')
# to get this working needed to do: sudo apt install cm-super
plt.rc("text", usetex = True)
plt.rc("figure", figsize = (textwidth, figheight))

s_none = np.load(args.dir + "sub_int_intensity_z_2210.npy")
s_sk = np.load(args.dir + "sub_int_intensity_z_sk_l1siguskmax_M256_2210.npy")
#s_sk = np.load(args.dir + "sub_int_intensity_z_sk_l4sigu4sig_M2048_2210.npy")
s_pt = np.load(args.dir + "sub_int_intensity_z_pt_2210.npy")

freq_ch = 921 #467 #280 #918 #208 #918
vmini = np.min(s_sk[:, freq_ch, :])
vmaxi = np.max(s_sk[:, freq_ch, :])


# identified ch 280 as an intermittent RFI ch
# wrt observation time:
# 45216 (total # pulses) * J0437_samples_T*time_resolution*10**-6 = 260 
# 260 / 60 = 4.3
num_sub_ints = s_sk.shape[0]
#num_2_roll = int(s_sk256.shape[2]/2 - list(s_sk256[:, freq_ch, :].sum(axis=0)).index(max(list(s_sk256[:, freq_ch, :].sum(axis=0)))))
#print("roll by: ", num_2_roll)

plt.figure(0)
#plt.imshow(np.roll(s_sk256[:, freq_ch, :], num_2_roll), origin="lower", aspect="auto", extent=[0, 1, 0, 4.3])
plt.imshow(s_sk[:, freq_ch, :], origin="lower", aspect="auto", extent=[0, 1, 0, 4.3], vmin=vmini, vmax=vmaxi)
func = lambda x, pos: "" if np.isclose(x,0) else x
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(func))
#plt.plot(s_sk[:, freq_ch, :].sum(axis=0), label="SK")
#plt.plot(s_none[:, freq_ch, :].sum(axis=0), label="none")
#plt.plot(s_pt[:, freq_ch, :].sum(axis=0), label="pt")
plt.xlabel("pulsar phase")
plt.ylabel("observation time [min]")
#plt.title("sk")
plt.savefig('/home/vereese/thesis_pics/sub_int_sk_l1siguskmax_M256_2210_ch' + str(freq_ch) + '.pdf', bbox_inches='tight')

plt.figure(1)
#plt.imshow(np.roll(s_none[:, freq_ch, :], num_2_roll), origin="lower", aspect="auto", extent=[0, 1, 0, 4.3])
plt.imshow(s_none[:, freq_ch, :], origin="lower", aspect="auto", extent=[0, 1, 0, 4.3], vmin=vmini, vmax=vmaxi)
func = lambda x, pos: "" if np.isclose(x,0) else x
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(func))
plt.xlabel("pulsar phase")
plt.ylabel("observation time [min]")
#plt.title("none")
plt.savefig('/home/vereese/thesis_pics/sub_int_none_2210_ch' + str(freq_ch) + '.pdf', bbox_inches='tight')

plt.figure(2)
#plt.imshow(np.roll(s_pt[:, freq_ch, :], num_2_roll), origin="lower", aspect="auto", extent=[0, 1, 0, 4.3])
plt.imshow(s_pt[:, freq_ch, :], origin="lower", aspect="auto", extent=[0, 1, 0, 4.3], vmin=vmini, vmax=vmaxi)
func = lambda x, pos: "" if np.isclose(x,0) else x
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(func))
plt.xlabel("pulsar phase")
plt.ylabel("observation time [min]")
#plt.title("pt")
plt.savefig('/home/vereese/thesis_pics/sub_int_pt_2210_ch' + str(freq_ch) + '.pdf', bbox_inches='tight')

textwidth = jai_textwidth
textheight = jai_textheight
font_size = jai_font
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
figheight = 0.3 * textwidth # used for JAI paper, the last plot only 
plt.rc('mathtext', fontset='cm')
# to get this working needed to do: sudo apt install cm-super
plt.rc("text", usetex = True)
plt.rc("figure", figsize = (textwidth, figheight))

fig, ax = plt.subplots(1, 3, sharey=True)
fig.tight_layout()
ax[0].imshow(s_none[:, freq_ch, :], origin="lower", aspect="auto", extent=[0, 1, 0, 4.3], vmin=vmini, vmax=vmaxi)
ax[1].imshow(s_pt[:, freq_ch, :], origin="lower", aspect="auto", extent=[0, 1, 0, 4.3], vmin=vmini, vmax=vmaxi)
ax[2].imshow(s_sk[:, freq_ch, :], origin="lower", aspect="auto", extent=[0, 1, 0, 4.3], vmin=vmini, vmax=vmaxi)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(func))
for i in np.arange(3):
    ax[i].set_xlabel("pulsar phase")
ax[0].set_ylabel("observation time [min]")
plt.savefig('/home/vereese/jai_pics/sub_int_2210_ch' + str(freq_ch) + '.pdf', bbox_inches='tight')
plt.show()
