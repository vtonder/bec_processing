import numpy as np
import matplotlib.pyplot as plt
from kurtosis import spectral_kurtosis
from constants import a4_textheight, a4_textwidth, thesis_font, jai_textwidth, jai_textheight, jai_font

'''
This script investigates the effect of clipping on spectral kurtosis. 
Read % clipped from last figure 2
'''

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

FFT_LEN = 1024
M = 512
mean = 0
stds = np.arange(10, 210, 10) / np.sqrt(2) # / np.sqrt(2) because want complex std 10 -> 200
print(stds[0])
print(stds[-1])

mean_sk = []
mean_sk_clipped = []
l_sk = []

clipped_std = []
l_clipped_std = []
total_clipped = []

# pretend FFT has already been taken, hence creating re, im data
for std in stds:
    wgn = np.random.normal(mean, std, size = M * FFT_LEN) + 1j * np.random.normal(mean, std, size = M * FFT_LEN)
    # clip at -127.5 and 126.5 because those are the decision levels and anything above that actually clips
    wgn_clipped = np.clip(wgn.real, -127.5, 126.5) + 1j*np.clip(wgn.imag, -127.5, 126.5)

    total_clipped.append(np.sum((wgn_clipped.real == 126.5) | (wgn_clipped.real == -127.5) | (wgn_clipped.imag == 126.5) | (wgn_clipped.imag == -127.5)))

    clipped_std.append(np.sqrt((np.var(wgn_clipped))))

    sk = spectral_kurtosis(wgn, M, FFT_LEN, fft=False, normalise=False)
    sk_clipped = spectral_kurtosis(wgn_clipped, M, FFT_LEN, fft=False, normalise=False)

    mean_sk.append(np.mean(sk))
    mean_sk_clipped.append(np.mean(sk_clipped))

    # Ludwig code
    x = np.clip(std * np.random.randn(FFT_LEN, M), -128, 127) + 1j * np.clip(std * np.random.randn(FFT_LEN, M), -128, 127)
    l_clipped_std.append(np.sqrt((np.var(x))))
    x2 = np.abs(x * x.conj())
    S1 = np.sum(x2, axis=-1)
    S2 = np.sum(x2 * x2, axis=-1)
    SK = M / (M - 1) * (M * S2 / (S1 * S1) - 1)
    l_sk.append(np.mean(SK))

print(clipped_std)
print(l_clipped_std)

plt.figure(0)
#plt.plot(stds, mean_sk, label="original", linewidth=2)
plt.axhline(1, color='m', linestyle = '-', linewidth=2, label="unclipped")
plt.plot(clipped_std, mean_sk_clipped, label="clipped", linewidth=2)
# plt.plot(l_clipped_std, l_sk, label="l clipping")
plt.axhline(0.77, color = 'g', linestyle = '--', linewidth=2, label="$\pm3\sigma$ thresholds")
plt.axhline(1.33, color = 'g', linestyle = '--', linewidth=2)
plt.vlines(x = 82, ymin=0 , ymax = 0.77, color= 'r', linestyle = '--', linewidth=2)
plt.legend()
plt.xlim([clipped_std[0], clipped_std[-1]])
plt.ylim([0.18, 1.4])
plt.xlabel('$\sigma_{\mbox{c}}$')
plt.ylabel('$ \overline{SK}$')
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/clip.pdf', bbox_inches='tight')

# Plot for JAI paper therefore reset some params
textwidth = jai_textwidth
textheight = jai_textheight
font_size = jai_font
plt.rc('font', size=font_size, family='serif')
plt.rc('axes', titlesize=font_size, labelsize=font_size)
plt.rc(('xtick', 'ytick'), labelsize=font_size)
plt.rc('legend', fontsize=font_size)

plt.figure(1)
plt.axhline(1, color='m', linestyle = '-', linewidth=2, label="unclipped")
plt.plot(clipped_std, mean_sk_clipped, label="clipped", linewidth=2)
plt.axhline(0.77, color = 'g', linestyle = '--', linewidth=2, label="$\pm3\sigma$ thresholds")
plt.axhline(1.33, color = 'g', linestyle = '--', linewidth=2)
plt.vlines(x = 82, ymin=0 , ymax = 0.77, color= 'r', linestyle = '--', linewidth=2)
plt.legend()
plt.xlim([clipped_std[0], clipped_std[-1]])
plt.ylim([0.18, 1.4])
plt.xlabel('$\sigma_{\mbox{c}}$')
plt.ylabel('$ \overline{SK}$')
plt.savefig('/home/vereese/Documents/PhD/jai-2e/clip.pdf', bbox_inches='tight')

# graph for percentages
plt.figure(2)
plt.plot(clipped_std, np.asarray(total_clipped)*100/(M*FFT_LEN))

plt.show()

'''
from scipy import signal, stats
# To add spikes - to pretend like they're pulsars (Code from Ludwig)
spike = np.random.rand(10000, M) > 0.99
ampl = np.where(spike, 200, 20)
x = np.clip(ampl * np.random.randn(10000, M), -127, 127) + 1j * np.clip(ampl * np.random.randn(10000, M), -127, 127)

# Generate a random signal with some narrowband and impulsive components
t = np.linspace(0, 1, 1000)
f1 = 50
f2 = 200
f3 = 400
x = np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t) + 0.1*np.sin(2*np.pi*f3*t)
x += 0.2*np.random.randn(len(x))

# Compute the spectral kurtosis of the original signal
f, Pxx = signal.welch(x, nperseg=256)
kurtosis_orig = stats.kurtosis(Pxx, fisher=False)

# Clip the signal between -0.5 and 0.5
x_clipped = np.clip(x, -0.5, 0.5)

# Compute the spectral kurtosis of the clipped signal
f, Pxx_clipped = signal.welch(x_clipped, nperseg=256)
kurtosis_clipped = stats.kurtosis(Pxx_clipped, fisher=False)

# Plot the original and clipped signals
fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
axs[0].plot(t, x)
axs[0].set_title('Original Signal')
axs[1].plot(t, x_clipped)
axs[1].set_title('Clipped Signal')
plt.xlabel('Time (s)')
plt.show()

# Print the spectral kurtosis of the original and clipped signals
print('Original spectral kurtosis:', kurtosis_orig)
print('Clipped spectral kurtosis:', kurtosis_clipped)'''
