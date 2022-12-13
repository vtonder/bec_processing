import numpy as np
import matplotlib.pyplot as plt

def spectral_kurtosis(s, M, FFT_LEN):

    perio = np.abs(np.fft.fft(s.reshape(M, FFT_LEN), axis=1)) ** 2 / FFT_LEN

    S1 = perio.sum(axis=0)
    S2 = np.sum(perio ** 2, axis=0)

    SK = ((M + 1) / (M - 1)) * ((M * S2 / S1 ** 2) - 1)

    return SK

if __name__ == "__main__":
    mean = 0
    std = 2
    FFT_LEN = 2048
    M = 6000 # number of averages to take PSD over
    N = FFT_LEN * M

    f1 = 10
    fs = 100
    t = np.arange(0, N/fs, 1.0/fs)
    s = np.sin(2*np.pi*f1*t)
    wgn = np.random.normal(mean, std, size=N)
    x = s + wgn

    SK = spectral_kurtosis(x, M, FFT_LEN)

    plt.figure(0)
    plt.axhline(1.6)
    plt.axhline(0.4)
    plt.plot(SK)
    plt.grid()
    plt.show()