import matplotlib.pyplot as plt
import numpy as np

#TODO need to apply normalisation to spectra otherwise the estimator will fail - Nita 2007
#row major implementation
def spectral_kurtosis(s, M, FFT_LEN, N = 1, d = 1, reshape= True, fft = True, normalise=True):

    if reshape:
        s = s.reshape(M, FFT_LEN)

    if fft:
        perio = np.abs(np.fft.fft(s, axis=1)) ** 2 / FFT_LEN
    else:
        perio = np.abs(s)**2 / FFT_LEN # FFT has already been taken

    #if normalise:
        #NTAPS = 8
        #PFB_M = NTAPS*FFT_LEN
        #wipfb_ndow = np.hamming(PFB_M) * np.sinc((np.arange(PFB_M) - PFB_M / 2.0) / FFT_LEN)
        #x = np.ones([M*FFT_LEN, FFT_LEN])
        #xf = np.zeros([M, FFT_LEN])
        #for i in np.arange(M):
        #    if (i + 8 == M - 1):
        #        break
        #    xf[i, :] = np.sum(np.multiply(x[i:i + 8, :], pfb_window), axis=0)
        #xf = np.sum(xf, axis=0)

        #perio = (2/(FFT_LEN*np.sum(pfb_window**2)))*perio

    S1 = perio.sum(axis=0) # like the mean of PSD u
    S2 = np.sum(perio ** 2, axis=0)

    SK = ((M*N*d + 1) / (M - 1)) * ((M * S2 / S1 ** 2) - 1)

    if normalise:
        pfb_window = np.hamming(FFT_LEN) * np.sinc((np.arange(FFT_LEN) - FFT_LEN / 2.0) / FFT_LEN)
        n = np.arange(FFT_LEN)
        k = np.arange(FFT_LEN)
        weights = np.ones(FFT_LEN)
        for i in k:
            for j in n:
                weights[i] += pfb_window[j]**2 * np.exp(-4*np.pi*i*j/FFT_LEN)
            weights[i] = weights[i]/np.sum(pfb_window**2)
        print("weights: ", np.abs(weights)**2)
        SK = SK-np.abs(weights)**2

    return SK

#column major SK implementation for use with filterbank data
def spectral_kurtosis_cm(s, M, FFT_LEN, N = 1, d = 1):
    perio = np.abs(s) ** 2 / FFT_LEN  # FFT has already been taken

    S1 = perio.sum(axis=1)
    S2 = np.sum(perio ** 2, axis=1)

    SK = ((M*N*d + 1) / (M - 1)) * ((M * S2 / S1 ** 2) - 1)

    return SK

#column major SK cross correlation implementation for use with filterbank data
def cc_spectral_kurtosis_cm(x1, x2, M, FFT_LEN, N = 1, d = 1):
    perio = np.abs(np.multiply(x1,np.conj(x2))) ** 2 / FFT_LEN  # FFT has already been taken

    S1 = perio.sum(axis=1)
    S2 = np.sum(perio ** 2, axis=1)

    SK = ((M*N*d + 1) / (M - 1)) * ((M * S2 / S1 ** 2) - 1)

    return SK

if __name__ == "__main__":
    mean = 0
    std = 2
    FFT_LEN = 1024
    M = 512 # number of averages to take PSD over
    N = FFT_LEN * M
    PFB_TAPS = 8
    PFB_M = PFB_TAPS*FFT_LEN

    f1 = 20
    fs = 100
    t = np.arange(0, N/fs, 1.0/fs)
    k = np.arange(FFT_LEN/2)
    f = k/FFT_LEN*fs
    s = np.sin(2*np.pi*f1*t)

    pulse_train = np.zeros(N)
    print("len pulse_train", len(pulse_train))
    print("number of samples", N)
    print("len pulse_train", pulse_train[2432000:2688000] )
    for i in np.arange(0,N,10*FFT_LEN):
        pulse_train[i+100:i+450] = np.ones(350)*50

    wgn = np.random.normal(mean, std, size=N)
    x =  wgn
    x1 =  wgn + pulse_train

    #xsk = gpt_spectral_kurtosis(x,1024)

    pfb_window = np.hamming(PFB_M) * np.sinc((np.arange(PFB_M) - PFB_M / 2.0) / FFT_LEN)
    pfb_window = pfb_window.reshape(PFB_TAPS, FFT_LEN)
    x = x.reshape(M, FFT_LEN)
    xf = np.zeros([M, FFT_LEN]) +1j*np.zeros([M, FFT_LEN])
    print("X shape: ", np.shape(x))
    for i in np.arange(M):
        if (i+PFB_TAPS == M-1):
            break
        a=x[i:i + PFB_TAPS, :] * pfb_window
        xf[i,:] = np.sum(a, axis=0)
        #xf[i,:] = np.sum(np.multiply(x[i:i+PFB_TAPS,:], pfb_window), axis=0)

    xf = xf[0:M-PFB_TAPS,:]
    XF = np.fft.fft(x, axis=1)
    XPF = np.fft.fft(xf, axis=1)

    PXF = np.abs(XF**2 / FFT_LEN).sum(axis=0)
    PXPF = np.abs(XPF**2 / FFT_LEN).sum(axis=0)

    SK1 = spectral_kurtosis(XF, M, FFT_LEN, reshape=False, fft=False, normalise=False)
    SK2 = spectral_kurtosis(XPF, M, FFT_LEN, reshape=False, fft=False, normalise=False)
    SK3 = spectral_kurtosis(XPF, M, FFT_LEN, reshape=False, fft=False, normalise=True)

    print("SK 1 mean:", np.mean(SK1))
    print("SK 2 mean:", np.mean(SK2))
    print("SK 3 mean:", np.mean(SK3))

    print("a")
    plt.figure(0)
    plt.plot(x.reshape(N,1))
    plt.grid()
    print("b")
    plt.figure(1)
    plt.plot(pfb_window.reshape(PFB_M))
    plt.grid()
    print("c")
    plt.figure(2)
    plt.plot(f, PXF[0:int(FFT_LEN/2)], label='FFT')
    plt.plot(f, PXPF[0:int(FFT_LEN/2)], label='PFB')
    plt.grid()
    plt.legend()
    print("d")
    plt.figure(3)
    plt.axhline(1.6)
    plt.axhline(0.4)
    plt.plot(f, SK1[0:int(FFT_LEN/2)], label='FFT No Norm')
    plt.plot(f, SK2[0:int(FFT_LEN/2)], label='PFB No Norm')
    plt.plot(f, SK3[0:int(FFT_LEN/2)], label='PFB Norm')
    plt.legend()
    plt.grid()

    #plt.figure(4)
    #plt.plot(xsk)
    #plt.grid()

    plt.show()

