import numpy as np
import time
from matplotlib import pyplot as plt




class bispectrum():
    """
    TODO:
    - verification with matlab / other scripts
    - No window is currently applied

    References:

    [1] A.P. Petropulu, Higher-Order Spectral Analysis, Drexel University
    [2] http://www1.maths.leeds.ac.uk/applied/news.dir/issue2/hos_intro.html#intro
    """
    def __init__(self, signal, max_lag, method='direct'):
        """
        :param signal: A stationary discrete time process. dimensions: K (records) x M (samples per record) numpy matrix
        :param max_lag: maximum lag in samples
        :param method: Either direct, which is the default or it can be set to indirect as defined in [1]
        :return:
        """
        self.signal = signal #
        self.max_lag = max_lag # ito samples not s
        self.K = int(signal.shape[0])
        self.M = int(signal.shape[1])
        self.power_spectrum = np.zeros([self.max_lag])
        self.bispectrum = np.zeros([self.max_lag, self.max_lag])
        self.method = method

    def mean_compensation(self):
        # calculate and subtract row mean ie mean of each record
        return self.signal - np.atleast_2d(np.mean(self.signal, axis=1)).transpose()

    def discrete_FT(self):
        N = len(self.signal)

        F = np.zeros([self.M,self.M], dtype='complex_')
        for k in np.arange(self.M):
            for n in np.arange(self.M):
                F[k,n] = k*n
        f_coef = np.exp((-1j*2*np.pi/self.M)*F)

        return np.dot(self.signal, f_coef)

    def calc_power_spectrum(self):

        S = self.discrete_FT()
        S = np.fft.fft(self.signal)
        P = np.abs(S)**2
        self.power_spectrum = 1.0/self.K * P.sum(axis=0)

        return self.power_spectrum

    def direct_bispectrum(self):

        # convert to frequency domain
        S = self.discrete_FT()
        S = np.fft.fft(self.signal)

        # calculate bispectrum on frequency data
        cum = np.zeros([self.K, self.max_lag, self.max_lag], dtype='complex_')
        for k1 in np.arange(self.max_lag):
            for k2 in np.arange(self.max_lag):
                cum[:, k1, k2] = (1.0/self.M) * S[:, k1] * np.conj(S[:, k2]) * np.conj(S[:,k1+k2])

        # return the average
        return 1.0/self.K * cum.sum(axis=0)

    def indirect_bispectrum(self):

        # calculate bispectrum on time data
        cum = np.zeros([K, self.max_lag, self.max_lag], dtype='complex_')
        for t1 in np.arange(self.max_lag):
            for t2 in np.arange(self.max_lag):
                for l in np.arange(self.max_lag):
                    cum[:,t1,t2] = cum[:,t1,t2] + self.signal[:,l]*self.signal[:,l+t1]*self.signal[:,l+t2]
                cum[:, t1, t2] = cum[:,t1,t2]/self.M #self.max_lag
    
        # calculate average
        avr_cum = (1.0/self.K) * cum.sum(axis=0)
    
        # calculate the fft of the cumulant which is the bispectrum
        return np.fft.fft2(avr_cum)

    def calc_bispectrum(self):
        # follow steps as per [1]
        self.mean_compensation()
        if self.method == 'direct':
            self.bispectrum = self.direct_bispectrum()
        else:
            self.bispectrum = self.indirect_bispectrum()

    def plot_power_spectrum(self, name=None):
        plt.figure(0)
        plt.plot(self.power_spectrum)
        #plt.plot(freq[0:500],bispectrum[:,])
        #plt.xticks(freq[0:500])
        plt.title(name)
        plt.show()

    def plot_bispectrum(self, name=None):
        plt.figure(0)
        #plt.plot(freq[0:500],bispectrum[:,])
        plt.imshow(np.abs(self.bispectrum), aspect='auto', origin='lower')
        #plt.xticks(freq[0:500])
        plt.title(name)
        plt.show()

if __name__=='__main__':
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
    noise = np.random.normal(0, 0.1, W) #+ s
    noise = noise.reshape(K, M)
    b = bispectrum(noise, M_2, method='direct')
    b.calc_bispectrum()
    b.calc_power_spectrum()
    b.plot_bispectrum()

"""plt.figure(0)
#plt.plot(freq[0:500],cum2[:,])
plt.imshow(np.abs(FN), aspect='auto', origin='lower') #, aspect='auto', extent=[0,500,0,500])
#plt.xticks(freq[0:500])

plt.figure(1)
#plt.plot(freq[0:500],cum2[:,])
plt.imshow(np.abs(cum2), aspect='auto', origin='lower') #, aspect='auto', extent=[0,500,0,500])
#plt.xticks(freq[0:500])
plt.show()"""