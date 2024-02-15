import numpy as np
from matplotlib import pyplot as plt
from constants import frequencies, a4_textheight, a4_textwidth, thesis_font

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

dpx = np.load("z1_231489536_2210_x.npy") # this is the number of dropped data points (ie both re and imag = 0)
dpy = np.load("z1_217726976_2210_y.npy")

dpx2 = np.load("z2_231489536_2210_x.npy") # this is the number of dropped data points (ie both re and imag = 0)
dpy2 = np.load("z2_217726976_2210_y.npy")


fig, ax = plt.subplots()
ax.plot(frequencies, (100*dpy) / 217726976, linewidth=2, label="V-polarisation")
ax.plot(frequencies, (100*dpx) / 231489536, linewidth=2, label="H-polarisation")
ax.set_xlim((frequencies[0], frequencies[-1]))
ax.set_ylim((0, 0.35))
ax.set_xlabel("frequency [MHz]")
ax.set_ylabel("\% complex zeros, one timesample")
plt.legend()
plt.grid()
plt.savefig('/home/vereese/thesis_pics/z1_data.pdf', transparent=True, bbox_inches='tight')

fig1, ax1 = plt.subplots()
ax1.plot(frequencies, (100*dpy2) / 217726976, linewidth=2, label="V-polarisation")
ax1.plot(frequencies, (100*dpx2) / 231489536, linewidth=2, label="H-polarisation")
ax1.set_xlim((frequencies[0], frequencies[-1]))
ax1.set_ylim((0, 0.021))
ax1.set_xlabel("frequency [MHz]")
ax1.set_ylabel("\% complex zeros, two timesamples")
plt.legend()
plt.grid()
plt.savefig('/home/vereese/thesis_pics/z2_data.pdf', transparent=True, bbox_inches='tight')

plt.show()

