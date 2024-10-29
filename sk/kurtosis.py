import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import sys
import matplotlib.ticker as ticker
sys.path.append('../')
from constants import a4_textheight, a4_textwidth, thesis_font, jai_textwidth, jai_textheight, jai_font

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

def spectral_kurtosis_cm(s, M, FFT_LEN, N = 1, d = 1):
    """
    - Column major SK implementation for use with filterbank data.
    - SK estimate is computed as per Equation 8 in "The generalized spectral kurtosis estimator" by Nita and Gary in 2010

    :param s: the complex signal with shape: number of frequency channels X M
    :param M: SK window size
    :param FFT_LEN: length of FFT
    :param N: correcting factor for conducting previous accumulations
    :param d: correcting factor for s not having Gamma distribution. Gamma distribution with 1, 2 degrees of freedom has d = 0.5, 1 respectively.
    :return: SK estimate with length: number of frequency channels
    """

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
    std_pulsar = 4
    FFT_LEN = 1024
    M = 512 # number of averages to take PSD over
    N = FFT_LEN * M
    PFB_TAPS = 8
    PFB_M = PFB_TAPS * FFT_LEN

    f1 = 40
    fs = 100
    t = np.arange(0, N / fs, 1.0 / fs)
    k = np.arange(FFT_LEN / 2)
    f = (k * fs) / FFT_LEN
    s = np.sin(2 * np.pi * f1 * t)

    pulse_train = np.zeros(N)
    duty_perc = 10
    duty_samples = int((duty_perc / 100) * M)

    print("len pulse_train", len(pulse_train))
    print("number of duty samples", duty_samples)

    pulsar = np.zeros(N)
    pulsar_duty = 10
    pulsar_duty_samples = int((pulsar_duty / 100) * M)
    print("on pulse samples: ", pulsar_duty_samples)

    for i, j in enumerate(np.arange(0, N, M)):
        pulse_train[j:j + duty_samples] = np.ones(duty_samples)*50 #np.random.randn(duty_samples) * 500  #
        if i % 5 == 0:
            pulsar[j:j + pulsar_duty_samples] = np.random.normal(mean, std_pulsar, size=pulsar_duty_samples) #+ 1j * np.random.normal(mean, std_pulsar, size=pulsar_duty_samples)

    #for i in np.arange(0, N, M):
    #    pulse_train[i:i+duty_samples] = np.random.randn(duty_samples)*500 #np.ones(duty_samples)*50

    wgn_re = np.random.normal(mean, std, size=N)
    wgn_im = np.random.normal(mean, std, size=N)

    x =  wgn_re #+ 1j*wgn_im
    x1 =  x + s
    x2 =  x + pulsar
    #x3 = x + pulse_train

    x = x.reshape(M, FFT_LEN)
    x1 = x1.reshape(M, FFT_LEN)
    x2 = x2.reshape(M, FFT_LEN)

    # NOTE: Adding 0s to the data raises the SK. Therefore, if you have dropped packets then you'll increase your SK
    #x[0:100,:] = 0
    print("N: ", N," x.shape: ", x.shape)
    print("% 0s:", 100*np.sum(np.where(x == 0, True, False))/N)

    #XF = np.fft.fft(x, axis=1)

    sk = spectral_kurtosis(x, M, FFT_LEN, reshape=False, fft=True, normalise=False)
    sk1 = spectral_kurtosis(x1, M, FFT_LEN, reshape=False, fft=True, normalise=False)
    sk2 = spectral_kurtosis(x2, M, FFT_LEN, reshape=False, fft=True, normalise=False)
    #print(np.mean(sk))
    #plt.figure(0)
    #plt.plot(x.flatten())

    # For thesis
    #plt.figure(1)
    #plt.plot(f[1:], sk[1:int(FFT_LEN / 2)], linewidth=2, label="$\overline{SK}$")
    #plt.xlim([f[1], f[-1]])
    #plt.axhline(0.77511, linestyle='--', color="g", linewidth=2, label="$\pm3\sigma$ thresholds")
    #plt.axhline(1.3254, linestyle='--', color="g", linewidth=2)
    #plt.axhline(np.mean(sk[1:int(FFT_LEN/2)]), color = "r", linestyle = '--', linewidth=2, label="mean of $\overline{SK}$")
    ##plt.ylim([0.65, 1.35])
    #plt.grid()
    #plt.xlabel("frequency [Hz]")
    #plt.ylabel('$\overline{SK}$')
    #plt.legend()
    ##plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/skest.pdf', bbox_inches='tight')
    ##plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/skest1.pdf', bbox_inches='tight')
    ##plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/skest2.pdf', bbox_inches='tight')
    #plt.show()

    textwidth = jai_textwidth
    textheight = jai_textheight
    font_size = jai_font
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
    figheight = 0.3 * textwidth  # used for JAI paper, the last plot only
    plt.rc('mathtext', fontset='cm')
    # to get this working needed to do: sudo apt install cm-super
    plt.rc("text", usetex=True)
    plt.rc("figure", figsize=(textwidth, figheight))

    sks = [sk, sk1, sk2]
    fig, ax = plt.subplots(1, 3, sharey=True)
    fig.tight_layout()


    #ax[1].plot(f[1:], sk1[1:int(FFT_LEN / 2)], linewidth=2)
    #ax[1].axhline(np.mean(sk1[1:int(FFT_LEN / 2)]), color="r", linestyle='--', linewidth=2)
    #ax[2].plot(f[1:], sk2[1:int(FFT_LEN / 2)], linewidth=2)
    #ax[2].axhline(np.mean(sk2[1:int(FFT_LEN / 2)]), color="r", linestyle='--', linewidth=2)
    func = lambda xval, pos: "" if np.isclose(xval, 0) else xval
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(func))
    for i, s in enumerate(sks):
        ax[i].set_xlim([f[1], f[-1]])
        ax[i].plot(f[1:], s[1:int(FFT_LEN / 2)], linewidth=2)
        ax[i].axhline(np.mean(s[1:int(FFT_LEN / 2)]), color="r", linestyle='--', linewidth=2)
        ax[i].set_xlabel("frequency [Hz]")
        ax[i].grid()
        #ax[i].set_ylim([0.65, 1.35])
        ax[i].axhline(0.77511, linestyle='--', color="g", linewidth=2)
        ax[i].axhline(1.3254, linestyle='--', color="g", linewidth=2)

    ax[0].set_ylabel("$\overline{SK}$")
    plt.savefig('/home/vereese/Documents/PhD/jai-2e/skest.pdf', bbox_inches='tight')
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



