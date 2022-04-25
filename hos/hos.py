import numpy as np
import time
from matplotlib import pyplot as plt


def sample(bi, w1, w2):
    offset = bi.shape[0] / 2
    in1 = int(w1 + offset)
    in2 = int(w2 + offset)
    return bi[in1, in2]


def write_b(bi, w1, w2, val):
    offset = bi.shape[0] / 2
    row = int(w1 + offset)
    col = int(w2 + offset)
    bi[row, col] = val


class Bispectrum():
    """
    TODO:
    - verification of frequency axis with matlab / other scripts
    - figure out scaling difference between indirect and direct methods
    - No window is currently applied
    - bichorency function
    - investigate quadratic phase coupling

    References:

    [1] A.P. Petropulu, "Higher-Order Spectral Analysis", Drexel University
    [2] http://www1.maths.leeds.ac.uk/applied/news.dir/issue2/hos_intro.html#intro
    [3] Appendix A.1 "The bispectrum and its relationship to phase-amplitude coupling"
    [4] CL Nikias and MR Raghuveer, "Bispectrum Estimation: A Digital Signal Processing Framework"
    """

    def __init__(self, signal, fft_size=1024, reshape=False, method='direct', fs=1):
        """
        :param signal: A stationary discrete time process.
        If reshape is False, then signal is a numpy matrix with dimensions: K (records) x M (samples per record)
        If reshape is True, then signal is assumed to be 1d.
        :param fft_size: Size of the FFT == the number of columns == M (samples per record)
        :param method: Either direct, which is the default or it can be set to indirect as defined in [1]
        :param reshape: the number of columns will be fft_size and the number of rows will be calculated accordingly
        :return:
        """
        self.signal = signal
        self.fft_size = fft_size
        self.M = int(fft_size)
        if reshape:
            data_len = len(list(signal))
            records = int(data_len/self.M)
            self.signal = np.asarray(self.signal[0:int(self.M * records)]).reshape(records, self.M)

        print("shape of signal", np.shape(self.signal))

        self.K = int(self.signal.shape[0])  # number of records
        self.max_lag = int(self.M / 2)  # ito samples not s
        self.power_spectrum = np.zeros([self.M])
        self.bispectrum_I = np.zeros([self.max_lag, self.max_lag])  # First (I) quadrant bispectrum
        self.full_bispec = np.zeros([self.M, self.M], dtype=np.csingle)
        self.bico_I = np.zeros([self.max_lag, self.max_lag], dtype=np.csingle)  # First (I) quadrant bicoherency
        #TODO: self.full_bico = np.zeros([self.M, self.M], dtype=np.csingle)
        self.method = method

        # frequencies
        self.fs = fs
        bw = fs / 2
        f_res = fs / self.M
        self.t_res = 1 / f_res
        self.freq = np.arange(-bw, bw, f_res)

        k = np.arange(-self.max_lag, self.max_lag)
        self.f = k / self.M
        self.w = 2*np.pi*self.f

        """k = np.arange(self.M)
        f = k/self.M"""

    def mean_compensation(self):
        # calculate and subtract row mean ie mean of each record
        self.signal = self.signal - np.atleast_2d(np.mean(self.signal, axis=1)).transpose()

        return self.signal

    def discrete_FT(self):

        F = np.zeros([self.M, self.M], dtype=np.csingle)
        for k in np.arange(self.M):
            for n in np.arange(self.M):
                F[k, n] = k * n
        f_coef = np.exp((-1j * 2 * np.pi / self.M) * F)

        return np.dot(self.signal, f_coef)

    def calc_power_spectrum(self):

        S = self.discrete_FT()
        S = np.fft.fft(self.signal)
        P = np.abs(S) ** 2
        self.power_spectrum = 1.0 / self.K * P.sum(axis=0)

        return self.power_spectrum

    def direct_bispectrum(self):

        # convert to frequency domain
        S = np.fft.fft(self.signal)

        # calculate bispectrum on frequency data
        cum = np.zeros([self.K, self.max_lag, self.max_lag], dtype=np.csingle)
        for k1 in np.arange(self.max_lag):
            for k2 in np.arange(self.max_lag):
                if k1 <= k2:
                    cum[:, k1, k2] = (1.0 / self.M) * S[:, k1] * S[:, k2] * np.conj(S[:, k1 + k2])
                    # cum[:, k2, k1] = cum[:, k1, k2] assignment also takes long only do this when want full bispectrum

        # average
        self.bispectrum_I = 1.0 / self.K * cum.sum(axis=0)

        return self.bispectrum_I


    def bicoherence(self):
        # TODO add a test to check if power spectrum and bispectrum has been calculated

        for k1 in np.arange(self.max_lag):
            for k2 in np.arange(self.max_lag):
                self.bico_I[k1,k2] = self.bispectrum_I[k1,k2]/np.sqrt(np.abs(self.power_spectrum[k1]*self.power_spectrum[k2]*self.power_spectrum[k1+k2]))

    def indirect_bispectrum(self):

        # calculate bispectrum on time data
        cum = np.zeros([K, self.max_lag, self.max_lag], dtype=np.csingle)
        for t1 in np.arange(self.max_lag):
            for t2 in np.arange(self.max_lag):
                for l in np.arange(self.max_lag):
                    cum[:, t1, t2] = cum[:, t1, t2] + self.signal[:, l] * self.signal[:, l + t1] * self.signal[:,l + t2]
                cum[:, t1, t2] = cum[:, t1, t2] / self.M  # self.max_lag

        # calculate average
        avr_cum = (1.0 / self.K) * cum.sum(axis=0)

        # calculate bispectrum which is the 2D fft of the cumulant
        self.bispectrum_I = np.fft.fft2(avr_cum)

        return self.bispectrum_I

    def calc_full_bispectrum(self):
        # follow steps as per [1] and implement symmetry as per [3]

        M_2 = int(self.M/2)
        print("before mean compensation", np.mean(self.signal))
        self.mean_compensation()
        print("after mean compensation", np.mean(self.signal))
        if self.method == 'direct':
            self.bispectrum_I = self.direct_bispectrum()
        else:
            self.bispectrum_I = self.indirect_bispectrum()

        # self.bispectrum_I = np.zeros([self.max_lag, self.max_lag], dtype=np.csingle)
        # self.bispectrum_I[100, 100:200] = 100000000
        # self.bispectrum_I[100:150, 200] = 100000000
        # self.bispectrum_I[100:200, 100] = 100000000
        # self.bispectrum_I[200, 100:150] = 100000000

        # quadrant I
        for k1 in np.arange(self.max_lag):
            for k2 in np.arange(self.max_lag):
                if k2 > k1:
                    self.bispectrum_I[k2, k1] = self.bispectrum_I[k1, k2]
        self.full_bispec[M_2:self.M, M_2:self.M] = self.bispectrum_I

        # quadrant III
        self.full_bispec[M_2:0:-1, M_2:0:-1] = self.bispectrum_I
        # quadrant II
        for w1 in np.arange(0, M_2):
            for w2 in np.arange(0, -M_2, -1):
                if w1 > -w2:
                    write_b(self.full_bispec, w1, w2, sample(self.full_bispec, -w1 - w2, w2))
                else:
                    write_b(self.full_bispec, w1, w2, sample(self.full_bispec, w1, -w1 - w2))
        # quadrant IV
        self.full_bispec[M_2:0:-1, M_2:self.M] = self.full_bispec[M_2:self.M, M_2:0:-1]

    def plot_power_spectrum(self, i, name=None):
        plt.figure(i)
        plt.plot(self.freq, self.power_spectrum)
        plt.title(name)
        plt.grid()
        #plt.show()

    def plot_bispectrum_I(self, name=None):

        plt.figure(0)
        plt.imshow(np.abs(self.bispectrum_I), aspect='auto', origin='lower')
        # plt.xticks(freq[0:500])
        plt.title(name)
        plt.show()

    def plot_full_bispectrum(self, name=None):

        plt.figure(0)
        plt.imshow(np.abs(self.full_bispec), aspect='auto', origin='lower', extent=([self.w[0],self.w[-1],self.w[0],self.w[-1]]))
        # plt.xticks(freq[0:500])
        plt.title(name)
        plt.show()

    def plot_bicoherence(self, name=None):

        plt.figure(0)
        plt.imshow(np.abs(self.bico_I), aspect='auto', origin='lower')
        plt.title(name)
        plt.show()

if __name__ == '__main__':
    # parameters
    f1 = 100  # Hz
    f2 = 250  # Hz
    f3 = f1 + f2  # Hz
    #phi1 = np.pi/3
    #phi2 = np.pi/4
    #phi3 = phi2+phi1
    fs = 1000  # Hz
    bw = fs / 2
    num_sec = 10
    t = np.arange(0, num_sec, 1 / fs)
    N = len(t)  # number of time samples
    W = N  # number of frequency samples
    K = int(10)  # number of records
    M = int(N / K)  # number of samples in a record
    M_2 = int(M / 2)
    f_res = fs / M
    t_res = 1 / f_res
    freq = np.arange(0, fs, f_res)

    s = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t) + np.cos(2 * np.pi * f3 *t)
    noise = np.random.normal(0, 0.1, W) + s
    noise = noise.reshape(K, M)
    b = Bispectrum(noise, fft_size=M, method='indirect', fs=1000)
    #b.calc_full_bispectrum()
    b.calc_power_spectrum()
    b.indirect_bispectrum()
    b.bicoherence()
    b.plot_bicoherence()
    #b.plot_power_spectrum()

    plt.figure(0)
    plt.imshow(np.abs(b.bispectrum_I),aspect='auto', origin='lower')
    plt.title('name')
    plt.show()
