import numpy as np
from pulsar_snr import PI
from matplotlib import pyplot as plt
import argparse
from constants import thesis_font, a4_textwidth, a4_textheight, jai_textwidth, jai_textheight, jai_font

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
figheight = 0.65 * textwidth
plt.rc('mathtext', fontset='cm')
# to get this working needed to do: sudo apt install cm-super
plt.rc("text", usetex = True)
plt.rc("figure", figsize = (textwidth, figheight))

#lt = ["0sig", "0_5sig","1sig", "2sig", "2_5sig", "3sig", "4sig", "skmin"] # lt : lower threshold
#ut = ["0sig", "0_5sig","1sig", "2sig", "2_5sig", "3sig", "4sig", "skmax"] # ut : upper threshold

lt = ["0sig", "0_5sig","1sig", "2sig", "2_5sig", "3sig", "4sig"] # lt : lower threshold
ut = ["0sig", "0_5sig","1sig", "2sig", "2_5sig", "3sig", "4sig"] # ut : upper threshold


M = ["64", "128", "256", "512", "1024", "2048", "4096", "8192"]

lt_snr = np.zeros([len(M), len(lt)])
ut_snr = np.zeros([len(M), len(ut)])

for i in np.arange(len(M)):
    # assume lower and upper thresholds have the same length
    print("\nMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM: ", M[i])

    for j in np.arange(len(lt)):
        # set initialise to False and call compute - this way no mask is applied
        print("")
        print("lt: ", lt[j])
        intensity_l = PI(args.dir, "sk_ypol_intensity_z_l" + lt[j] + "_M" + M[i] + "_m1_n1_1064_p144.npy", "sk_ypol_num_nz_z_l" + lt[j] + "_M" + M[i] + "_m1_n1_1064_p144.npy")
        #intensity_l = PI(args.dir, "sk_intensity_z_l" + lt[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy", "sk_num_nz_z_l" + lt[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy", initialise = False)
        #intensity_l.compute()

        print("")
        print("ut: ", ut[j])
        intensity_u = PI(args.dir, "sk_ypol_intensity_z_u" + ut[j] + "_M" + M[i] + "_m1_n1_1064_p144.npy", "sk_ypol_num_nz_z_u" + ut[j] + "_M" + M[i] + "_m1_n1_1064_p144.npy")
        #intensity_u = PI(args.dir, "sk_intensity_z_u" + ut[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy", "sk_num_nz_z_u" + ut[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy")
        #intensity_u = PI(args.dir, "sk_intensity_z_u" + ut[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy", "sk_num_nz_z_u" + ut[j] + "_M" + M[i] + "_m1_n1_2210_p45216.npy", initialise = False)
        #intensity_u.compute()

        lt_snr[i,j] = intensity_l.snr
        ut_snr[i,j] = intensity_u.snr

vmini = np.min(ut_snr) 
vmaxi = np.max(lt_snr) 

lt_labels = ("$0\sigma$", "$0.5\sigma$", "$1\sigma$", "$2\sigma$", "$2.5\sigma$", "$3\sigma$", "$4\sigma$") #, "$SK_{min}$")
x_range = np.arange(len(lt_labels))
y_range = np.arange(len(M))
#fig, ax = plt.subplots()
#snr_im_lt = ax.imshow(lt_snr, origin="lower", aspect="auto", vmin=vmini, vmax=vmaxi)
#cbar = fig.colorbar(snr_im_lt, ax=ax)
#cbar.minorticks_on()
#ax.set_ylabel("$M$")
#ax.set_xlabel("Lower threshold")
#ax.set_xticks(x_range, lt_labels)
#ax.set_yticks(y_range, M)
#plt.savefig('/home/vereese/thesis_pics/lower_threshold_masked.pdf', transparent=True, bbox_inches='tight')

ut_labels = ("$0\sigma$", "$0.5\sigma$", "$1\sigma$", "$2\sigma$", "$2.5\sigma$", "$3\sigma$", "$4\sigma$") #, "$SK_{max}$")
#fig1, ax1 = plt.subplots()
#snr_im_ut = ax1.imshow(ut_snr, origin="lower", aspect="auto", vmin=vmini, vmax=vmaxi)
#cbar1 = fig1.colorbar(snr_im_ut, ax=ax1)
#cbar1.minorticks_on()
#ax1.set_xticks(x_range, ut_labels)
#ax1.set_yticks(y_range, M)
#ax1.set_ylabel("$M$")
#ax1.set_xlabel("Upper threshold")
#plt.savefig('/home/vereese/thesis_pics/upper_threshold_masked.pdf', transparent=True, bbox_inches='tight')

# Plot for JAI paper therefore reset some params
textwidth = jai_textwidth
textheight = jai_textheight
font_size = jai_font
plt.rc('font', size=font_size, family='serif')
plt.rc('axes', titlesize=font_size, labelsize=font_size)
plt.rc(('xtick', 'ytick'), labelsize=font_size)
plt.rc('legend', fontsize=font_size)
figheight = 0.4 * textwidth
plt.rc("figure", figsize = (textwidth, figheight))

fig2, ax2 = plt.subplots(1, 2, sharey=True)
fig2.tight_layout()
snr_im_ltut = ax2[0].imshow(lt_snr, origin="lower", aspect="auto", vmin=vmini, vmax=vmaxi)
snr_im_ltut = ax2[1].imshow(ut_snr, origin="lower", aspect="auto", vmin=vmini, vmax=vmaxi)
ax2[0].set_xticks(x_range, lt_labels)
ax2[0].set_yticks(y_range, M)
ax2[0].set_ylabel("$M$")
ax2[0].set_xlabel("Lower threshold")
ax2[1].set_xticks(x_range, ut_labels)
ax2[1].set_yticks(y_range, M)
ax2[1].set_xlabel("Upper threshold")
fig2.subplots_adjust()
cbar2 = fig2.colorbar(snr_im_ltut, ax=ax2)
cbar2.minorticks_on()
#plt.savefig('/home/vereese/jai_pics/lower_upper_threshold_masked_vela.pdf', transparent=True, bbox_inches='tight')

plt.show()
