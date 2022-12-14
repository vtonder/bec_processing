import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

#TODO need to apply normalisation to spectra otherwise the estimator will fail - Nita 2007
def spectral_kurtosis(s, M, FFT_LEN, fft = True, normalise=True):
    N = FFT_LEN * M
    if fft:
        perio = np.abs(np.fft.fft(s.reshape(M, FFT_LEN), axis=1)) ** 2 / FFT_LEN
    else:
        perio = s.reshape(M, FFT_LEN) # FFT has already been taken

    if normalise:
        NTAPS = 8
        PFB_M = NTAPS*FFT_LEN
        pfb_window = np.hamming(PFB_M) * np.sinc((np.arange(PFB_M) - PFB_M / 2.0) / FFT_LEN)
        perio = 2*perio/(N*np.sum(pfb_window**2))

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

    f1 = 20
    fs = 100
    t = np.arange(0, N/fs, 1.0/fs)
    k = np.arange(FFT_LEN/2)
    f = k/FFT_LEN*fs
    s = np.sin(2*np.pi*f1*t)
    wgn = np.random.normal(mean, std, size=N)
    x = s + wgn
    coeffs = firwin(128, [0.05, 0.25], width=0.05, pass_zero=False)
    h_even = coeffs[::2]
    h_odd = coeffs[1::2]

    y_even = lfilter(h_even, 1, x)
    y_odd = lfilter(h_odd, 1, x)

    # Reconstruct the filtered signal
    y = y_even + y_odd


    SK = spectral_kurtosis(y, M, FFT_LEN)

    plt.figure(0)
    plt.axhline(1.6)
    plt.axhline(0.4)
    plt.plot(f, SK[0:int(FFT_LEN/2)])
    plt.grid()
    plt.show()