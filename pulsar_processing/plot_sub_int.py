import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
from constants import thesis_font, a4_textwidth, a4_textheight

# This script assists in calculating the drift in the pulse period
# It uses eq. 7.5 from the handbook
# sub_int_vela_11.1946499395.npy plots the sub integration over 4 vela pulse phases
# It was produced using the 1604641569_wide_tied_array_channelised_voltage_0x.h5 file
# A sub-integration happens over 22 vela pulses, this number was randomly chosen
# deltaT in samples = 211375
# 11.1946499395*(1-(211375/252755968))

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

DIR = '/home/vereese/git/phd_data/pulsar/'
PREFIX = 'sub_int_vela_11.'
files = ['18500936838522.npy', '185031494489326.npy', '185053620637202.npy', '185075.npy', '18512.npy', '1946499395.npy']
vela_sub_int = np.load(DIR+PREFIX+files[2])
last = vela_sub_int[-1,:]
first = vela_sub_int[8,:] #np.roll(np.asarray(vela_sub_int[8,:]),-700)

plt.figure(0)
plt.ylabel('observation time [min]')
plt.xlabel('pulse phase')
plt.imshow(vela_sub_int, aspect='auto', extent=[0, 1, 0, 4.3], origin='lower')
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/subintcor.pdf', transparent=True, bbox_inches='tight')

plt.figure(1)
plt.plot(last, label='last')
plt.plot(first, label='first')
plt.grid()
plt.legend()

plt.show()