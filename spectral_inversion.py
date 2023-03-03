import matplotlib.pyplot as plt
import numpy as np

# frequency units Hz
f1 = np.arange(1,6)*10.0
w1 = 2*np.pi*f1

fs = 100.0
Ts = 1/fs

t = np.arange(0, 1, Ts)
N = K = len(t)
f = np.arange(-fs/2,fs/2,fs/K)
inverter = np.asarray(int(N/2)*[1,-1])
x = np.zeros(N)
for i, w in enumerate(w1, start=1):
    x = x + i*np.cos(w*t)

xi = x*inverter

plt.figure(0)
plt.plot(t,x)
plt.xlabel('time s')
plt.grid()

plt.figure(1)
plt.plot(f,np.fft.fftshift(np.fft.fft(x)), label='original')
plt.plot(f,np.fft.fftshift(np.fft.fft(xi)), label='inverted')
plt.xlabel('frequency Hz')
plt.legend()
plt.grid()
plt.show()