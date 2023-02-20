import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from kurtosis import spectral_kurtosis_cm, spectral_kurtosis

'''
This script investigates the effect of clipping on spectral kurtosis
'''

FFT_LEN = 1024
M = 100000
mean = 0
std = 1

wgn_raw = np.random.normal(mean, std, size=M*FFT_LEN)
wgn_cm = np.fft.fft(wgn_raw.reshape([FFT_LEN, M]), axis=1)

sk_cm = spectral_kurtosis_cm(wgn_cm, M, FFT_LEN)
sk_rm = spectral_kurtosis(wgn_raw, M, FFT_LEN, fft=True, normalise=False)

print("sk_rm", np.mean(sk_rm))
print("sk_cm", np.mean(sk_cm))

#plt.figure()
#plt.plot(sk_cm[0:int(FFT_LEN/2)], label='CM')
#plt.plot(sk_rm[0:int(FFT_LEN/2)], label='RM')
#plt.grid()
#plt.legend()
#plt.axhline(0.77, linestyle = '--')
#plt.axhline(1.33, linestyle = '--')
#plt.show()

'''# Generate a random signal with some narrowband and impulsive components
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