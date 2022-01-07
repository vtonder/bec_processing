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


class bispectrum():
    """
    TODO:
    - verification of frequency axis with matlab / other scripts
    - figure out scaling difference between indirect and direct methods
    - No window is currently applied

    References:

    [1] A.P. Petropulu, Higher-Order Spectral Analysis, Drexel University
    [2] http://www1.maths.leeds.ac.uk/applied/news.dir/issue2/hos_intro.html#intro
    [3] Appendix A.1 "The bispectrum and its relationship to phase-amplitude coupling"
    """

    def __init__(self, signal, method='direct', fs=1):
        """
        :param signal: A stationary discrete time process. dimensions: K (records) x M (samples per record) numpy matrix
        :param method: Either direct, which is the default or it can be set to indirect as defined in [1]
        :return:
        """
        self.signal = signal  #
        self.K = int(signal.shape[0])
        self.M = int(signal.shape[1]) # number of samples == FFT size
        self.max_lag = int(self.M / 2)  # ito samples not s
        self.power_spectrum = np.zeros([self.M])
        self.bispectrum_I = np.zeros([self.max_lag, self.max_lag])  # First (I) quadrant bispectrum
        self.full_bispec = np.zeros([self.M, self.M], dtype='complex_')
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
        return self.signal - np.atleast_2d(np.mean(self.signal, axis=1)).transpose()

    def discrete_FT(self):

        F = np.zeros([self.M, self.M], dtype='complex_')
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
        cum = np.zeros([self.K, self.max_lag, self.max_lag], dtype='complex_')
        for k1 in np.arange(self.max_lag):
            for k2 in np.arange(self.max_lag):
                cum[:, k1, k2] = (1.0 / self.M) * S[:, k1] * S[:, k2] * np.conj(S[:, k1 + k2])

        # return the average
        return 1.0 / self.K * cum.sum(axis=0)

    def indirect_bispectrum(self):

        # calculate bispectrum on time data
        cum = np.zeros([K, self.max_lag, self.max_lag], dtype='complex_')
        for t1 in np.arange(self.max_lag):
            for t2 in np.arange(self.max_lag):
                for l in np.arange(self.max_lag):
                    cum[:, t1, t2] = cum[:, t1, t2] + self.signal[:, l] * self.signal[:, l + t1] * self.signal[:,l + t2]
                cum[:, t1, t2] = cum[:, t1, t2] / self.M  # self.max_lag

        # calculate average
        avr_cum = (1.0 / self.K) * cum.sum(axis=0)

        # calculate bispectrum which is the 2D fft of the cumulant
        return np.fft.fft2(avr_cum)

    def calc_bispectrum(self):
        # follow steps as per [1] and implement symmetry as per [3]

        M_2 = int(self.M/2)
        self.mean_compensation()
        if self.method == 'direct':
            self.bispectrum_I = self.direct_bispectrum()
        else:
            self.bispectrum_I = self.indirect_bispectrum()

        # self.bispectrum_I = np.zeros([self.max_lag, self.max_lag], dtype='complex_')
        # self.bispectrum_I[100, 100:200] = 100000000
        # self.bispectrum_I[100:150, 200] = 100000000
        # self.bispectrum_I[100:200, 100] = 100000000
        # self.bispectrum_I[200, 100:150] = 100000000

        # quadrant I
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

    def plot_power_spectrum(self, name=None):
        plt.figure(0)
        plt.plot(self.freq, self.power_spectrum)
        plt.title(name)
        plt.grid()
        plt.show()

    def plot_bispectrum(self, name=None):

        plt.figure(0)
        plt.imshow(np.abs(self.full_bispec), aspect='auto', origin='lower', extent=([self.w[0],self.w[-1],self.w[0],self.w[-1]]))
        # plt.xticks(freq[0:500])
        plt.title(name)
        plt.show()


if __name__ == '__main__':
    # parameters
    f1 = 100  # Hz
    f2 = 250  # Hz
    f3 = f1 + f2  # Hz
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

    s = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t) + np.cos(2 * np.pi * f3 * t)
    noise = np.random.normal(0, 0.1, W) + s
    noise = noise.reshape(K, M)
    b = bispectrum(noise, method='direct', fs=1000)
    b.calc_bispectrum()
    b.calc_power_spectrum()
    b.plot_bispectrum()
    b.plot_power_spectrum()

    plt.figure(0)
    plt.plot(freq, b.power_spectrum)
    plt.title('name')
    plt.grid()
    plt.show()
