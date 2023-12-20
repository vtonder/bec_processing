import random
import numpy as np
from kurtosis import spectral_kurtosis_cm
from matplotlib import pyplot as plt
from constants import a4_textheight, a4_textwidth, thesis_font

"""
This script investigates the effect of zeros in the data on the SK.
It was found that adding zeros to the data set raises the SK.
Therefore, if you have dropped packets in your data then you'll SK will be more that increase your SK
"""

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

M = 512
num_sk = 10000
num_dp = M * num_sk # number of data points
perc = np.arange(0, 0.2, 0.01)
#print("N: ", N, " x.shape: ", x.shape)
print("% 0s:", 100 * np.sum(np.where(perc == 0, True, False)) / (M * num_sk))

#XF = np.fft.fft(x, axis=1)
mean_sk = []
for p in perc:
    # randn generates noise from the standard Gaussian distribution ie mean = 0; std = 1 = var
    x = np.random.randn(num_dp) + 1j * np.random.randn(num_dp)
    ndpz = int(num_sk * M * p) # number of data points to zero
    indices = random.sample(range(num_dp - 1), ndpz)
    x[indices] = 0
    x = x.reshape(num_sk, M)
    mean_sk.append(np.mean(spectral_kurtosis_cm(x, M, 1024)))

plt.figure()
plt.plot(perc*100, (np.asarray(mean_sk) - 1)*100, linewidth=2)
plt.xlabel("\% dropped packets")
plt.ylabel("\% increase in $\overline{SK}$")
plt.xlim([perc[0]*100, perc[-1]*100])
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/drop.pdf', bbox_inches='tight')
plt.show()