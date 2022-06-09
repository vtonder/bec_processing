from matplotlib import pyplot as plt
import numpy as np

def shift_spectrum(S):
    K = len(S)
    K_2= int(K/2)
    S_shifted = np.zeros(K, dtype=S.dtype)
    print("CASR", S.dtype)
    S_shifted[0:K_2] = S[K_2:K]
    S_shifted[K_2:K] = S[0:K_2]

    return S_shifted

def dft(x, K):
    N = len(x)
    print(N)
    num_col = K
    num_row = int(N / K)
    F = np.zeros([K, K], dtype='complex_')
    for k in np.arange(K):
        for n in np.arange(K):
            F[k, n] = k * n
    f_coef = np.exp((-1j * 2 * np.pi / K) * F)
    X = np.sum(np.dot(x[0:num_row*num_col].reshape(num_row, num_col), f_coef), axis=0)/np.sqrt(num_row)
    print(np.shape(X))
    return X

f1 = 100.0  # Hz
f2 = 200.0  # Hz
fs = 1000.0 # Hz
bw = fs/2
num_sec = 1
t = np.arange(0, num_sec, 1/fs)
N = len(t)

K = int(512) # fft length
K_2 = int(K/2) # fft length / 2
upsample = 10 # specific for pulse train

freq_res = fs/K
freq = np.arange(0,fs,freq_res)
freq2 = np.arange(-fs/2,fs/2,freq_res)

cos = np.cos(2*np.pi*f1*t) + np.cos(2*np.pi*f2*t)
s = np.random.random(N) # white noise ie samples are uncorrelated from each other
s2 = np.random.normal(0, 0.1, N) # Gaussian process

pulse_train = np.round(np.random.random(N))*2 - 1  # array*2 - 1 is to get alternating -1 1
pulse_train_up = pulse_train.reshape(N, 1)
for i in np.arange(upsample):
    pulse_train_up = np.concatenate((pulse_train_up, pulse_train.reshape(N,1)), axis=1)
pulse_train_up = pulse_train_up.flatten()
x = cos

X = np.fft.fft(x, K)
S = np.abs(X)**2
S_shifted = shift_spectrum(S)

X2 = dft(x, K)
S2 = np.abs(X2)**2
print("TYPE", S2.dtype)
S2_shifted = shift_spectrum(S2)

plt.figure(0)
#plt.plot(np.arange(M), Self, label='dft')
#plt.plot(s, label='white')
#plt.plot(s2, label='gaussian')
plt.plot(pulse_train, label='pulse_train')
plt.plot(pulse_train_up, label='pulse_train_up')
plt.xlabel("time samples n")
plt.legend()
plt.grid()

plt.figure(1)
plt.plot(freq2, X, label='fft')
plt.plot(freq2, X2, label='dft')
#plt.xlabel("frequency samples k")
plt.xlabel("frequency Hz")
plt.legend()
plt.grid()

plt.figure(2)
plt.plot(freq2, S_shifted, label='fft')
plt.plot(freq2, S2_shifted, label='dft')
#plt.xlabel("frequency samples k")
plt.xlabel("frequency Hz")
plt.legend()
plt.grid()

plt.figure(3)
plt.plot(freq2, S2_shifted, label='dft')
#plt.xlabel("frequency samples k")
plt.xlabel("frequency Hz")
plt.legend()
plt.grid()

plt.show()


