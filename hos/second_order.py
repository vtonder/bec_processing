from matplotlib import pyplot as plt
import numpy as np

f1 = 100.0  # Hz
f2 = 200.0  # Hz
fs = 1000.0 # Hz
bw = fs/2
num_sec = 3
t = np.arange(0, num_sec, 1/fs)
N = len(t)
M = N
freq_res = fs/M
freq = np.arange(0,fs,freq_res)

s = np.cos(2*np.pi*f1*t) + np.cos(2*np.pi*f2*t)
#s = np.cos(2*np.pi*f1*t) #+ np.cos(2*np.pi*f2*t)
N = len(s)

F = np.zeros([N, N], dtype='complex_')
for k in np.arange(N):
    for n in np.arange(N):
        F[k, n] = k * n
f_coef = np.exp((-1j * 2 * np.pi / N) * F)

Self = np.dot(s, f_coef)

S = np.fft.fft(s)

plt.figure(0)
plt.plot(freq, Self, label='dft')
plt.plot(freq,S, label='fft')
plt.legend()
plt.grid()
plt.show()


