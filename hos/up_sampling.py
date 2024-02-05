import numpy as np
from hos import Bispectrum
from matplotlib import pyplot as plt
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

fft_size = 2048
N = 100 * fft_size
# up_sample_factor = 5 need to make this generic

pulse_train = np.round(np.random.random(N)) * 2 - 1  # array*2 - 1 is to get alternating -1 1
pulse_train_up = pulse_train.reshape(N, 1)
pulse_train_up = np.concatenate((pulse_train_up, pulse_train_up, pulse_train_up, pulse_train_up, pulse_train_up), axis=1)
pulse_train_up = list(pulse_train_up.flatten())

b = Bispectrum(pulse_train, reshape=True, fft_size=fft_size, method='direct')
b_up = Bispectrum(pulse_train_up, reshape=True, fft_size=fft_size, method='direct')
# b = Bispectrum(pulse_train, reshape=True, fft_size=1024, method='direct', fs=1000)
b.calc_full_bispectrum()
b_up.calc_full_bispectrum()

b_ps = b.calc_power_spectrum()
bup_ps = b_up.calc_power_spectrum()
#b.direct_bispectrum()

# b.plot_full_bispectrum(i=1)

plt.figure(0)
plt.plot(pulse_train[0:102], label='pulse train')
plt.plot(pulse_train_up[0:102], label=r'$\uparrow$ pulse train')
plt.xlim(0,100)
plt.ylabel("Polar NRZ")
plt.xlabel("time samples n")
plt.legend()
plt.grid()
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/pulse_trains.pdf', transparent=True, bbox_inches='tight')
plt.savefig('/home/vereese/thesis_pics/pulse_trains.pdf', transparent=True, bbox_inches='tight')

k = np.arange(-1024,1024)
#k = np.arange(-512,512)
plt.figure(1)
plt.plot(k, b_ps,  label='pulse train')
plt.plot(k, bup_ps, label=r'$\uparrow$ pulse train')
plt.xlim(-1024,1024)
plt.ylabel("PSD of line code")
plt.xlabel("frequency samples k")
plt.legend()
plt.grid()
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/psd_pt.pdf', transparent=True, bbox_inches='tight')
plt.savefig('/home/vereese/thesis_pics/psd_pt.pdf', transparent=True, bbox_inches='tight')

plt.figure(2)
plt.imshow(np.abs(b.full_bispec), aspect='auto', origin='lower',extent=([-1024, 1024, -1024, 1024]))
plt.xlabel("frequency samples k")
plt.ylabel("frequency samples k")
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/bispec_pt.pdf', transparent=True, bbox_inches='tight')
plt.savefig('/home/vereese/thesis_pics/bispec_pt.pdf', transparent=True, bbox_inches='tight')

plt.figure(3)
plt.imshow(np.abs(b_up.full_bispec), aspect='auto', origin='lower',extent=([-1024, 1024, -1024, 1024]))
plt.xlabel("frequency samples k")
plt.ylabel("frequency samples k")
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/bispec_pt_up.pdf', transparent=True, bbox_inches='tight')
plt.savefig('/home/vereese/thesis_pics/bispec_pt_up.pdf', transparent=True, bbox_inches='tight')

plt.show()

'''plt.figure(2)
plt.hist(pulse_train)
plt.title("Pulse train")

plt.figure(3)
plt.hist(pulse_train_up.flatten())
plt.title("Pulse train up")'''