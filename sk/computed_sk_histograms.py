import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
from constants import frequencies, lower_limit_4s, lower_limit_1s, upper_limit_skmax, upper_limit_4s, a4_textwidth, a4_textheight, thesis_font, frequencies

# A script to plot the histograms of computed SK's of real data

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

#M = [64, 256]
M = [256]
pdf = {}
sk = {}
for m in M:
    pdf.update({m:np.load("pdf_M" + str(m) + ".npy")})
    sk.update({m:np.load("sk_z_M" + str(m) + "_m1_n1_2210_0y.npy")})

# This x range comes from sk_thresholds which generated the pdf's
x = np.arange(0, 2, 0.01)
ch_names = ["clipped gsm", "dme", "gal e5a", "gal e5b", "gps l2", "glonass", "clean", "iridium", "iridium"]
chs = [105, 280, 383, 420, 445, 467, 600, 918, 921]

m = 256
l_1s = np.round(lower_limit_1s[m], 2)
u_sm = np.round(upper_limit_skmax[m], 2)
l_4s = np.round(lower_limit_4s[m], 2)
u_4s = np.round(upper_limit_4s[m], 2)

"""
# clipped GSM at 943.77 MHz 
ch = 105 
fig0, ax0 = plt.subplots()
ax0.semilogy(x, pdf[m], label="Pearson type IV PDF", linewidth=2)
ax0.hist(sk[m][ch, :], 1000, density = "True", stacked = "True", log = "True")
ax0.grid()
ax0.set_ylabel("SK PDF")
ax0.set_ylim([10**-4, 10])
ax0.set_xlim([0, 2])
ax0.set_xlabel("SK values")
#ax0.set_title("M: " + str(m) + ", ch: " + str(ch) + "clipped gsm ")
ax0.vlines(x=l_4s, ymin = 10**-4, ymax = 10, color='r', label="4$\sigma$ thresholds", linewidth=2)
ax0.vlines(x=u_4s, ymin = 10**-4, ymax = 10, color='r', linewidth=2)
ax0.vlines(x=l_1s, ymin = 10**-4, ymax = 10, color='g', label="1$\sigma$ lower threshold", linewidth=2)
#ax0.vlines(x=u_sm, ymin = 10**-4, ymax = 10, color='m', label="$SK_{max}$ upper threshold", linewidth=2)
ax0.legend()
plt.savefig('/home/vereese/thesis_pics/sk_hist_m256_ch105.pdf', transparent=True, bbox_inches='tight')

# DME ch 
ch = 280 
fig1, ax1 = plt.subplots()
ax1.semilogy(x, pdf[m], label="Pearson type IV PDF", linewidth=2)
ax1.hist(sk[m][ch, :], 1000, density = "True", stacked = "True", log = "True")
ax1.grid()
ax1.set_ylabel("SK PDF")
ax1.set_ylim([10**-4, 10])
ax1.set_xlim([0.5, 11])
ax1.set_xlabel("SK values")
#ax1.set_title("M: " + str(m) + ", ch: " + str(ch) + "dme")
ax1.vlines(x=l_4s, ymin = 10**-4, ymax = 10, color='r', label="4$\sigma$ thresholds", linewidth=2)
ax1.vlines(x=u_4s, ymin = 10**-4, ymax = 10, color='r', linewidth=2)
ax1.vlines(x=l_1s, ymin = 10**-4, ymax = 10, color='g', label="1$\sigma$ lower threshold", linewidth=2)
ax1.vlines(x=u_sm, ymin = 10**-4, ymax = 10, color='m', label="$SK_{max}$ upper threshold", linewidth=2)
ax1.legend(loc='upper center')
plt.savefig('/home/vereese/thesis_pics/sk_hist_m256_ch' + str(ch) + '.pdf', transparent=True, bbox_inches='tight')

# glonass at 1246.38 MHz
ch = 467 
fig2, ax2 = plt.subplots()
ax2.semilogy(x, pdf[m], label="Pearson type IV PDF", linewidth=2)
ax2.hist(sk[m][ch, :], 1000, density = "True", stacked = "True", log = "True")
ax2.grid()
ax2.set_ylabel("SK PDF")
ax2.set_ylim([10**-4, 10])
ax2.set_xlim([0.5, 2])
ax2.set_xlabel("SK values")
#ax2.set_title("M: " + str(m) + ", ch: " + str(ch) + "glonass")
ax2.vlines(x=l_4s, ymin = 10**-4, ymax = 10, color='r', label="4$\sigma$ thresholds", linewidth=2)
ax2.vlines(x=u_4s, ymin = 10**-4, ymax = 10, color='r', linewidth=2)
ax2.vlines(x=l_1s, ymin = 10**-4, ymax = 10, color='g', label="1$\sigma$ lower threshold", linewidth=2)
#ax2.vlines(x=u_sm, ymin = 10**-4, ymax = 10, color='m', label="$SK_{max}$ upper threshold", linewidth=2)
ax2.legend()
plt.savefig('/home/vereese/thesis_pics/sk_hist_m256_ch' + str(ch) + '.pdf', transparent=True, bbox_inches='tight')

# H1 at 1420.4 
ch = 675  
fig3, ax3 = plt.subplots()
ax3.semilogy(x, pdf[m], label="Pearson type IV PDF", linewidth=2)
ax3.hist(sk[m][ch, :], 1000, density = "True", stacked = "True", log = "True")
ax3.grid()
ax3.set_ylabel("SK PDF")
ax3.set_ylim([10**-4, 10])
ax3.set_xlim([0.5, 2])
ax3.set_xlabel("SK values")
#ax3.set_title("M: " + str(m) + ", ch: " + str(ch) + "h1")
ax3.vlines(x=l_4s, ymin = 10**-4, ymax = 10, color='r', label="4$\sigma$ thresholds", linewidth=2)
ax3.vlines(x=u_4s, ymin = 10**-4, ymax = 10, color='r', linewidth=2)
ax3.vlines(x=l_1s, ymin = 10**-4, ymax = 10, color='g', label="1$\sigma$ lower threshold", linewidth=2)
#ax3.vlines(x=u_sm, ymin = 10**-4, ymax = 10, color='m', label="$SK_{max}$ upper threshold", linewidth=2)
ax3.legend()
plt.savefig('/home/vereese/thesis_pics/sk_hist_m256_ch' + str(ch) + '.pdf', transparent=True, bbox_inches='tight')
"""

# iridium at 1625.9
ch = 921
fig4, ax4 = plt.subplots()
ax4.semilogy(x, pdf[m], label="Pearson type IV PDF", linewidth=2)
ax4.hist(sk[m][ch, :], 1000, density = "True", stacked = "True", log = "True")
ax4.grid()
ax4.set_ylabel("SK PDF")
ax4.set_ylim([10**-4, 10])
ax4.set_xlim([0, 2])
ax4.set_xlabel("SK values")
#ax4.set_title("M: " + str(m) + ", ch: " + str(ch) + "iridium")
ax4.vlines(x=l_4s, ymin = 10**-4, ymax = 10, color='r', label="4$\sigma$ thresholds", linewidth=2)
ax4.vlines(x=u_4s, ymin = 10**-4, ymax = 10, color='r', linewidth=2)
ax4.vlines(x=l_1s, ymin = 10**-4, ymax = 10, color='g', label="1$\sigma$ lower threshold", linewidth=2)
#ax4.vlines(x=u_sm, ymin = 10**-4, ymax = 10, color='m', label="$SK_{max}$ upper threshold", linewidth=2)
ax4.legend(loc="upper right")
plt.savefig('/home/vereese/thesis_pics/sk_hist_m256_ch' + str(ch) + '.pdf', transparent=True, bbox_inches='tight')

plt.show()


