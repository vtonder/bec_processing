import numpy as np
from matplotlib import pyplot as plt

font = {'family': 'STIXGeneral',
        'size': 42}
plt.rc('font', **font)

M = 512
duty_cycles = np.arange(100)

duty_samples = (duty_cycles / 100) * M
wgn = np.random.normal(0, 1, size=M) + 1j*np.random.normal(0, 1, size=M)
snrs = [1,2, 5,10,50]
SK = [[], [], [], [], []]
#S1 = []
#S2 = []
for i, snr in enumerate(snrs):
    for dc in duty_samples:
        s = np.zeros([M])
        s[0:int(dc)] = snr
        s = wgn+s
        perio = np.abs(s)**2
        S1 = perio.sum()  # like the mean of PSD u
        S2 = np.sum(perio ** 2)
        SK[i].append(((M + 1) / (M - 1)) * ((M * S2 / S1 ** 2) - 1))

plt.figure(0, figsize=[22,16])
plt.plot(duty_cycles, SK[0], label="SNR=1")
plt.plot(duty_cycles, SK[1], label="SNR=2")
plt.plot(duty_cycles, SK[2], label="SNR=5")
plt.plot(duty_cycles, SK[3], label="SNR=10")
plt.plot(duty_cycles, SK[4], label="SNR=50")
plt.axhline(y=0.77511, xmin=0,xmax=100, linestyle='--')
plt.axhline(y=1.3254, xmin=0, xmax=100, linestyle='--')
plt.grid()
plt.ylabel("Spectral Kurtosis")
plt.ylim([-1,4])
plt.xlim([0,95])
plt.xlabel("Duty Cycle %")
plt.legend()
plt.savefig('/home/vereese/Documents/PhD/jai-2e/duty.eps', bbox_inches='tight')
plt.show()