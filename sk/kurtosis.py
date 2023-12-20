import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from constants import a4_textheight, a4_textwidth, thesis_font

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

# column major SK implementation for use with filterbank data
# when power (periodogram) is given to function
def spectral_kurtosis_cm_perio(perio, M, N = 1, d = 1):
    S1 = perio.sum(axis=1)
    S2 = np.sum(perio ** 2, axis=1)

    SK = ((M*N*d + 1) / (M - 1)) * ((M * S2 / S1 ** 2) - 1)

    return SK

def s1_s2(s, FFT_LEN):
    perio = np.abs(s) ** 2 / FFT_LEN  # FFT has already been taken

    S1 = perio.sum(axis=1)
    S2 = np.sum(perio ** 2, axis=1)

    return S1, S2

#multiscale column major SK implementation for use with filterbank data
def ms_spectral_kurtosis_cm(S1, S2, M, N = 1, d = 1, m = 1, n = 1):

    M = M*n*m
    kernel = np.ones((n, m))

    ns1 = signal.convolve2d(S1, kernel, mode="valid")
    ns2 = signal.convolve2d(S2, kernel, mode="valid")
    
    SK = ((M*N*d + 1) / (M - 1)) * ((M * ns2 / ns1 ** 2) - 1)

    return SK

def complex_spectral_kurtosis(s, M):

    S1 = ((np.abs(s))**2).sum(axis=1)
    S2 = (np.real(s)**4 + np.imag(s)**4).sum(axis=1)
    SK = 2*M*S2/S1**2

    return SK - 3

#column major SK cross correlation implementation for use with filterbank data
def cc_spectral_kurtosis_cm(x1, x2, M, FFT_LEN, N = 1, d = 1):
    perio = np.abs(np.multiply(x1,np.conj(x2))) ** 2 / FFT_LEN  # FFT has already been taken

    S1 = perio.sum(axis=1)
    S2 = np.sum(perio ** 2, axis=1)

    SK = ((M*N*d + 1) / (M - 1)) * ((M * S2 / S1 ** 2) - 1)

    return SK

# Equation 13 from Nita 2016 titled: "Spectral Kurtosis statistics of transient signals"
def sk_gaus_tran(pho, delta):
    offset = (2 * (1 - delta) * delta * (pho**2))/((1 + (delta * pho))**2)

    return 1 + offset

if __name__ == "__main__":
    textwidth = a4_textwidth
    textheight = a4_textheight
    font_size = thesis_font
    # groups are like plt.figure plt.legend etc
    plt.rc('font', size=font_size, family='serif')
    plt.rc('pdf', fonttype=42)
    # plt.rc('axes', titlesize=14, labelsize=14)
    plt.rc('axes', titlesize=font_size, labelsize=font_size)
    plt.rc(('xtick', 'ytick'), labelsize=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('lines', markersize=5)
    # The following should only be used for beamer
    # plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
    figheight = 0.65 * textwidth
    plt.rc('mathtext', fontset='cm')
    # to get this working needed to do: sudo apt install cm-super
    plt.rc("text", usetex=True)
    plt.rc("figure", figsize=(textwidth, figheight))

    mean = 0
    std = 1
    FFT_LEN = 1024
    M = 512 # number of averages to take PSD over
    N = FFT_LEN * M
    PFB_TAPS = 8
    PFB_M = PFB_TAPS*FFT_LEN

    f1 = 40
    fs = 100
    t = np.arange(0, N/fs, 1.0/fs)
    k = np.arange(FFT_LEN/2)
    f = k/FFT_LEN*fs
    s = np.sin(2*np.pi*f1*t)

    pulse_train = np.zeros(N)
    duty_perc = 80
    duty_samples = int((duty_perc/100)*M)

    print("len pulse_train", len(pulse_train))
    print("number of duty samples", duty_samples)

    for i in np.arange(0, N, M):
        pulse_train[i:i+duty_samples] = np.random.randn(duty_samples)*500 #np.ones(duty_samples)*50

    wgn_re = np.random.normal(mean, std, size=N)
    wgn_im = np.random.normal(mean, std, size=N)

    x =  wgn_re #+ 1j*wgn_im
    x =  x + s #pulse_train
    x = x.reshape(M, FFT_LEN)

    # NOTE: Adding 0s to the data raises the SK. Therefore, if you have dropped packets then you'll increase your SK
    #x[0:100,:] = 0
    print("N: ", N," x.shape: ", x.shape)
    print("% 0s:", 100*np.sum(np.where(x == 0, True, False))/N)

    XF = np.fft.fft(x, axis=1)

    sk = spectral_kurtosis(x, M, FFT_LEN, reshape=False, fft=True, normalise=False)
    print(np.mean(sk))
    plt.figure(0)
    plt.plot(x.flatten())

    plt.figure(1)
    plt.plot(f[1:], sk[1:int(FFT_LEN/2)], linewidth=2)
    plt.xlim([f[1], f[-1]])
    plt.axhline(0.77511, linestyle = '--', linewidth=2, label="thresholds")
    plt.axhline(1.3254, linestyle = '--', linewidth=2)
    #plt.ylim([0.65, 1.35])
    plt.grid()
    plt.xlabel("frequency [Hz]")
    plt.ylabel('$\overline{SK}$')
    plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/skest1.pdf', bbox_inches='tight')
    plt.show()

    '''x = x.reshape(FFT_LEN, M)
    XF = np.fft.fft(x, axis=0)
    SK = complex_spectral_kurtosis(XF, M)
    print("SK shape: ", SK.shape)

    x = x.transpose()
    XF = np.fft.fft(x, axis=1)
    SK1 = spectral_kurtosis(XF, M, FFT_LEN, reshape=False, fft=False, normalise=False)

    plt.figure(0)
    plt.plot(SK)
    plt.ylabel("SK")
    plt.xlabel("frequency")
    plt.grid()

    plt.figure(1)
    plt.plot(SK1)
    plt.ylabel("SK")
    plt.xlabel("frequency")
    plt.grid()

    plt.show()

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

    x1 = x.reshape(FFT_LEN, M)
    XF = np.fft.fft(x1, axis=0)
    SK = complex_spectral_kurtosis(XF, M)
    print("SK shape: ", SK.shape)

    plt.figure(0)
    plt.plot(f, SK[int(FFT_LEN/2):], label='FFT No Norm')
    plt.ylabel("SK")
    plt.xlabel("frequency")
    plt.grid()

    #xf = xf[0:M-PFB_TAPS,:]
    x = x.reshape(M, FFT_LEN)
    XF = np.fft.fft(x, axis=1)
    #XPF = np.fft.fft(xf, axis=1)

    #PXF = np.abs(XF**2 / FFT_LEN).sum(axis=0)
    #PXPF = np.abs(XPF**2 / FFT_LEN).sum(axis=0)

    SK1 = spectral_kurtosis(XF, M, FFT_LEN, reshape=False, fft=False, normalise=False)
    #SK2 = spectral_kurtosis(XPF, M, FFT_LEN, reshape=False, fft=False, normalise=False)
    #SK3 = spectral_kurtosis(XPF, M, FFT_LEN, reshape=False, fft=False, normalise=True)

    print("SK 1 mean:", np.mean(SK1))
    #print("SK 2 mean:", np.mean(SK2))
    #print("SK 3 mean:", np.mean(SK3))
    plt.figure(1)
    plt.plot(f, SK1[0:int(FFT_LEN/2)], label='FFT No Norm')
    plt.ylabel("SK")
    plt.xlabel("frequency")
    plt.grid()

    plt.show()
    plt.figure(0)
    plt.plot(x.reshape(N,1))
    plt.grid()
    plt.figure(1)
    plt.plot(pfb_window.reshape(PFB_M))
    plt.grid()
    plt.figure(2)
    plt.plot(f, PXF[0:int(FFT_LEN/2)], label='FFT')
    plt.plot(f, PXPF[0:int(FFT_LEN/2)], label='PFB')
    plt.grid()
    plt.legend()
    plt.figure(3)
    plt.axhline(1.6)
    plt.axhline(0.4)
    plt.plot(f, SK1[0:int(FFT_LEN/2)], label='FFT No Norm')
    plt.plot(f, SK2[0:int(FFT_LEN/2)], label='PFB No Norm')
    plt.plot(f, SK3[0:int(FFT_LEN/2)], label='PFB Norm')
    plt.legend()
    plt.grid()
    plt.show()
    '''



