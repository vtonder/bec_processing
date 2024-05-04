import numpy as np
from matplotlib import pyplot as plt
from constants import a4_textwidth, a4_textheight, thesis_font
from pulsar_processing.pulsar_functions import incoherent_dedisperse

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

vela = np.load("/home/vereese/git/phd_data/pulsar/summed_profile_1234_0x.npy")
#vela = incoherent_dedisperse(vela,"1569")

for i in np.arange(1024):
    vela[i,:] = vela[i,:] - np.mean(vela[i,:])

plt.imshow(vela, origin="lower", aspect="auto")
plt.show()
#vela_top = vela[512:,:].sum(axis=0)
#vela_bottom = vela[0:512].sum(axis=0)

# profile length
#vl = vela.shape[1]
# index where maximum value occurs
#vmi = vela_profile.argmax()
#vm = np.max(vela_profile)
#vela_centered = np.roll(vela_profile, int((vl/2)-vmi))
#vela_floor = np.mean(vela_centered[0:1000])
plt.show()