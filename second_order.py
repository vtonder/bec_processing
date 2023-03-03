from matplotlib import pyplot as plt
import numpy as np

# Upsample by increasing fs - pushes aliases further away
# Sample and hold pulse train - spectrum becomes sync envelope. convolve pulse train with a sync (ft of pulse), see ch7.2.1 of Lathi

font = {'family': 'STIXGeneral',
        'size': 26}
plt.rc('font', **font)

def shift_spectrum(S):
    K = len(S)
    K_2= int(K/2)
    S_shifted = np.zeros(K, dtype=S.dtype)
    print("CASR", S.dtype)
    S_shifted[0:K_2] = S[K_2:K]
    S_shifted[K_2:K] = S[0:K_2]

    return S_shifted

def dft(x):
    K = len(x)
    print(N)
    num_col = K
    num_row = int(N / K)
    F = np.zeros([K, K], dtype='complex_')
    for k in np.arange(K):
        for n in np.arange(K):
            F[k, n] = k * n
    f_coef = np.exp((-1j * 2 * np.pi / K) * F)
    #X = np.sum(np.dot(x[0:num_row*num_col].reshape(num_row, num_col), f_coef), axis=0)/np.sqrt(num_row)
    X = np.dot(x, f_coef)
    print(np.shape(X))
    return X

f1 = 100.0  # Hz
f2 = 200.0  # Hz
fs = 1000.0 # Hz
bw = fs/2
num_sec = 5
t = np.arange(0, num_sec, 1/fs)
N = len(t)
print(N)
K = int(1024) # fft length
K_2 = int(K/2) # fft length / 2
upsample = 10 # specific for pulse train

freq_res = fs/K  # For FFT
freq_res2 = fs/(N*(upsample+1)) # For DFT
freq = np.arange(0,fs,freq_res)
freq2 = np.arange(0,fs,freq_res2)
freq_shifted = np.arange(-fs/2,fs/2,freq_res)
freq_shifted2 = np.arange(-fs/2,fs/2,freq_res2)

cos = np.cos(2*np.pi*f1*t) + np.cos(2*np.pi*f2*t)
s = np.random.random(N) # white noise ie samples are uncorrelated from each other
s2 = np.random.normal(0, 0.1, N) # Gaussian process

top_hat = np.zeros(N)
top_hat[0:int(3*N/4)] = 1
#top_hat = top_hat*2 -1

pulse_train = np.round(np.random.random(N))*2 - 1  # array*2 - 1 is to get alternating -1 1
pulse_train_up = pulse_train.reshape(N, 1)
for i in np.arange(upsample):
    pulse_train_up = np.concatenate((pulse_train_up, pulse_train.reshape(N,1)), axis=1)
pulse_train_up = pulse_train_up.flatten()

x = pulse_train

X = np.fft.fft(x, K)
S = np.abs(X)**2
S_shifted = shift_spectrum(S)

x = pulse_train_up

X = np.fft.fft(x, K)
S2 = np.abs(X)**2
S2_shifted = shift_spectrum(S2)
print(np.shape(S2_shifted))
plt.figure(0, figsize=[20,20])
#plt.plot(np.arange(M), Self, label='dft')
#plt.plot(s, label='white')
#plt.plot(s2, label='gaussian')
#plt.plot(top_hat)
plt.plot(pulse_train, label='line code')
plt.plot(pulse_train_up, label='up-sampled line code')
plt.xlabel("time samples n")
plt.legend()
plt.grid()

plt.figure(1, figsize=[20,20])
plt.plot(freq_shifted, S_shifted, label='PSD line code')
plt.plot(freq_shifted, S2_shifted, label='PSD up-sampled line code')
plt.xlabel("frequency samples k")
#plt.xlabel("frequency Hz")
plt.legend()
plt.xlim([-int(K/2), int(K/2)])
plt.grid()
plt.savefig('/home/vereese/Documents/PhD/thesis/Figures/upsampled', bbox_inches='tight')

x = np.linspace(-np.pi, np.pi, 100)
sinc = np.sinc(x)*np.sinc(x) # this is the upsampled case
sinc2 = np.sinc(0.1*x)*np.sinc(0.1*x) # this is the original case
plt.figure(2, figsize=[20,20])
plt.plot(x, sinc, label='f(x)=sinc(x)')
plt.plot(x, sinc2, label='f(x)=sinc(x/10)')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.xlim([-np.pi, np.pi])
plt.legend()
plt.grid()
plt.savefig('/home/vereese/Documents/PhD/thesis/Figures/sinc', bbox_inches='tight')

plt.show()


