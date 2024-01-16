import numpy as np
from matplotlib import pyplot as plt
from constants import frequencies, a4_textheight, a4_textwidth, thesis_font

textwidth = a4_textwidth
textheight = a4_textheight
font_size = thesis_font

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

dpx = np.load("dp_231489536_2210_x.npy") # this is the number of dropped data points (ie both re and imag = 0)
dpy = np.load("dp_217726976_2210_y.npy")

fig, ax = plt.subplots()
ax.plot(frequencies, (100*dpy) / 217726976, linewidth=2, label="Y-pol")
ax.plot(frequencies, (100*dpx) / 231489536, linewidth=2, label="X-pol")
ax.set_xlim((frequencies[0], frequencies[-1]))
ax.set_xlabel("frequency [MHz]")
ax.set_ylabel("\% dropped data points")
plt.legend()
plt.grid()
plt.savefig('/home/vereese/thesis_pics/dropped_data.pdf', transparent=True, bbox_inches='tight')
plt.show()

