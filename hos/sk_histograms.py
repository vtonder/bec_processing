import numpy as np
from kurtosis import spectral_kurtosis

mean = 0
std = 2
FFT_LEN = 1024
M = 512  # number of averages to take PSD over
N = FFT_LEN * M

SK = np.zeros([FFT_LEN, 300000])
for i in np.arange(300000):
    wgn = np.random.normal(mean, std, size=N)
    x = wgn
    SK[:,i] = spectral_kurtosis(x, M, FFT_LEN, normalise=False)

np.save("SK_histograms", SK)