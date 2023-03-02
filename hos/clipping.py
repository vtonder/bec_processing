import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from kurtosis import spectral_kurtosis_cm, spectral_kurtosis

'''
This script investigates the effect of clipping on spectral kurtosis
'''

FFT_LEN = 1024
M = 1000
mean = 0
stds = np.arange(10, 200, 10)

mean_sk = []
mean_sk_clipped = []
l_sk = []

clipped_std = []
l_clipped_std = []

# pretend fft has already been taken, hence creating re, im data
for std in stds:
    wgn = np.random.normal(mean, std, size = M * FFT_LEN) + 1j * np.random.normal(mean, std, size = M * FFT_LEN)
    wgn_clipped = np.clip(wgn.real, -127, 127) + 1j*np.clip(wgn.imag, -127, 127)

    clipped_std.append(np.sqrt((np.var(wgn_clipped))))

    sk = spectral_kurtosis(wgn, M, FFT_LEN, fft=False, normalise=False)
    sk_clipped = spectral_kurtosis(wgn_clipped, M, FFT_LEN, fft=False, normalise=False)

    mean_sk.append(np.mean(sk))
    mean_sk_clipped.append(np.mean(sk_clipped))

    # Ludwig code
    #x = np.clip(std * np.random.randn(10000, M), -127, 127) + 1j * np.clip(std * np.random.randn(10000, M), -127, 127)
    x = np.clip(std * np.random.randn(FFT_LEN, M), -127, 127) + 1j * np.clip(std * np.random.randn(FFT_LEN, M), -127, 127)
    l_clipped_std.append(np.sqrt((np.var(x))))
    x2 = np.abs(x * x.conj())
    S1 = np.sum(x2, axis=-1)
    S2 = np.sum(x2 * x2, axis=-1)
    SK = M / (M - 1) * (M * S2 / (S1 * S1) - 1)
    l_sk.append(np.mean(SK))

print(clipped_std)
print(l_clipped_std)

plt.figure()
#plt.plot(sk[0:int(FFT_LEN/2)], label='Row Major')
plt.plot(stds, mean_sk, label="no clipping")
plt.plot(clipped_std, mean_sk_clipped, label="v clipping")
plt.plot(l_clipped_std, l_sk, label="l clipping")
plt.grid()
plt.legend()
plt.axhline(0.77, linestyle = '--')
plt.axhline(1.33, linestyle = '--')
plt.xlabel('respective stds')
plt.ylabel('mean SK')
plt.show()

# Ludwig code:

lstd = 30


# To add spikes
spike = np.random.rand(10000, M) > 0.99
ampl = np.where(spike, 200, 20)
x = np.clip(ampl * np.random.randn(10000, M), -127, 127) + 1j * np.clip(ampl * np.random.randn(10000, M), -127, 127)

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