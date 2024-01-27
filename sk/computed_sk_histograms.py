import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
from constants import a4_textwidth, a4_textheight, thesis_font, frequencies

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

for i, ch in enumerate(chs):
    fig_offset = i*len(M)
    for j, m in enumerate(M):
        plt.figure(fig_offset + j)
        plt.semilogy(x, pdf[m])
        plt.hist(sk[m][ch, :], 1000, density = "True", stacked = "True", log = "True")
        plt.grid()
        plt.ylabel("SK PDF")
        plt.ylim([10**-4, 10**2])
        plt.xlabel("SK")
        plt.title("M: " + str(m) + ", ch: " + str(ch) + " " + ch_names[i])
plt.show()


