import matplotlib.pyplot as plt
import numpy as np
from constants import a4_textwidth, a4_textheight, thesis_font

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

# frequency units Hz
f1 = np.arange(1,6)*10.0
w1 = 2*np.pi*f1

fs = 100.0
Ts = 1/fs

t = np.arange(0, 1, Ts)
N = K = len(t)
f = np.arange(-fs/2,fs/2,fs/K)
inverter = np.asarray(int(N/2)*[1,-1])
x = np.zeros(N)
for i, w in enumerate(w1, start=1):
    x = x + i*np.cos(w*t)

xi = x*inverter

plt.figure(0)
plt.plot(t,x)
plt.xlabel('time s')
plt.grid()

plt.figure(1)
plt.plot(f,np.fft.fftshift(np.fft.fft(x)), label='original', linewidth=2)
plt.plot(f,np.fft.fftshift(np.fft.fft(xi)), label='inverted', linewidth=2)
plt.xlabel('frequency [Hz]')
plt.ylabel('X(n)')
plt.legend()
plt.xlim([-50, 49])
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/spec_inver.pdf', bbox_inches='tight')
plt.show()