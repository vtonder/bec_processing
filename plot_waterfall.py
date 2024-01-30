import numpy as np
from matplotlib import pyplot as plt
import argparse
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

data = np.load("/home/vereese/data/phd_data/waterfall_1569_0x_1638400.npy")
maxi = np.max(data) / 4
mini = np.min(data)

plt.figure()
plt.imshow(data, origin="lower", aspect="auto", extent=[0, 4.3, 856, 1712], vmin=mini, vmax=maxi)
plt.xlabel("observation time [min]")
plt.ylabel("frequency [MHz]")
plt.savefig('/home/vereese/thesis_pics/waterfall.pdf', bbox_inches='tight')
#plt.show()
