import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from constants import thesis_font, a4_textwidth, a4_textheight, beamer_textwidth, beamer_textheight, beamer_font

#DIR_OUT = "/home/vereese/Documents/PhD/ThesisTemplate/Figures/"
#DIR_OUT = "/home/vereese/Documents/PhD/URSI2023/"
DIR_OUT = "/home/vereese/thesis_pics/"
#DIR_OUT = "/home/vereese/presentation_pics/"

#DIR_IN = "/home/vereese/git/phd_data/sk_analysis/ursi_data/"
DIR_IN = "/home/vereese/git/phd_data/sk_analysis/ursi_data/"

# Setup fonts and sizes for publication, based on page dimensions in inches
textwidth = a4_textwidth
textheight = a4_textheight
font_size = thesis_font
#textwidth = beamer_textwidth
#textheight = beamer_textheight
#font_size = beamer_font

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

p1 = np.load(DIR_IN+'ursi_profile.npy')
p2 = np.load(DIR_IN+'ursi_sf_lu.npy')
p3 = np.load(DIR_IN+'ursi_sf_l.npy')
p4 = np.load(DIR_IN+'ursi_sf_l2.npy')

mini = min(p2.flatten())
maxi = max(p2.flatten())/4

plt.figure(0)
plt.imshow(p1, aspect='auto', extent=[0, 1, 856, 1712] , origin='lower', vmin=mini, vmax=maxi)
plt.xlabel('pulse phase')
plt.ylabel('frequency [MHz]')
plt.colorbar()
#plt.title('Vela Pulse Profile without RFI mitigation')
#plt.savefig(DIR_OUT+'p1.pdf', bbox_inches='tight')

plt.figure(1)
plt.imshow(p2, aspect='auto', extent=[0, 1, 856, 1712], origin='lower', cmap='gray_r', vmin=mini, vmax=maxi)
plt.xlabel('pulse phase')
plt.ylabel('frequency [MHz]')
plt.colorbar(cmap='gray_r')
#plt.title('(b) Upper and lower thresholds are applied, M=512')
#plt.savefig(DIR_OUT+'p2.pdf', bbox_inches='tight')

plt.figure(2)
plt.imshow(p3, aspect='auto', extent=[0, 1, 856, 1712] , origin='lower', cmap='gray_r', vmin=mini, vmax=maxi)
#plt.xlim([26.8, 62.58])
plt.xlabel('pulse phase')
plt.ylabel('frequency [MHz]')
plt.colorbar(cmap='gray_r')
#plt.title('(c) Only lower thresholds are applied, M=512')
#plt.savefig(DIR_OUT+'p3.pdf', bbox_inches='tight')

plt.figure(3)
plt.imshow(p4, aspect='auto', extent=[0, 1, 856, 1712] , origin='lower', cmap='gray_r', vmin=mini, vmax=maxi)
#plt.xlim([26.8, 62.58])
plt.xlabel('pulse phase')
plt.ylabel('frequency [MHz]')
plt.colorbar(cmap='gray_r')
#plt.title('(d) Only lower thresholds are applied, M=2048')
#plt.savefig(DIR_OUT+'p4.pdf', bbox_inches='tight')

plt.show()
