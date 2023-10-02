import matplotlib.pyplot as plt
import numpy as np

textwidth = 9.6 # 128.0 / 25.4 #
textheight = 7 # 96.0 / 25.4 # 7
plt.rc('font', size=12, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=12, labelsize=12)
plt.rc(('xtick', 'ytick'), labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

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
plt.plot(f,np.fft.fftshift(np.fft.fft(x)), label='original', linewidth=2)
plt.plot(f,np.fft.fftshift(np.fft.fft(xi)), label='inverted', linewidth=2)
plt.xlabel('frequency [Hz]')
plt.ylabel('X(n)')
plt.legend()
plt.xlim([-50, 49])
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/spec_inver.eps', bbox_inches='tight')
plt.show()