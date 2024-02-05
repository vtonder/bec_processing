from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append("../")
from constants import a4_textwidth, a4_textheight, thesis_font

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

phi1 = -2*np.pi
phi2 = 2*np.pi
N = 1000
steps = np.pi / N
range1 = np.arange(phi1,phi2, steps)

a = np.sinc(range1)
b = np.sinc(range1/10)

plt.figure()
plt.plot(range1, b, label="sinc$(x/10)$")
plt.plot(range1, a, label="sinc$(x)$")
plt.xlabel("$x$ values")
plt.ylabel("$f(x)$")
plt.xlim([range1[0], range1[-1]])
plt.legend()
plt.grid()
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/sinc.pdf', transparent=True, bbox_inches='tight')
plt.show()
