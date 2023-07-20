import numpy as np
from matplotlib import pyplot as plt

#font = {'family': 'STIXGeneral',
#        'size': 42}
#plt.rc('font', **font)

textwidth = 9.6 #128.0 / 25.4
textheight = 7 #96.0 / 25.4
plt.rc('font', size=22, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=22, labelsize=22)
plt.rc(('xtick', 'ytick'), labelsize=22)
plt.rc('legend', fontsize=22)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

M = 512
#duty_cycles = np.array([20,30,40,50,80,80,50,80,10,80,80,80,80,10,50,80,40,10,80,50,90,10,20,30,5,50,20,80])
duty_cycles = np.arange(0,110)
msklen = len(duty_cycles) - 4
print("len dc: ", len(duty_cycles))
print("len msklen: ", msklen )
duty_samples = (duty_cycles / 100) * M
wgn = np.random.normal(0, 1, size=M) + 1j*np.random.normal(0, 1, size=M)
snrs = [1,2, 5,10,50]
SK = [[], [], [], [], []]
S1 = []
S2 = []
for i, snr in enumerate(snrs):
    S1.append([])
    S2.append([])
    for j, dc in enumerate(duty_samples):
        s = np.zeros([M])
        s[0:int(dc)] = snr
        s = wgn+s
        perio = np.abs(s)**2
        S1[i].append(perio.sum())  # like the mean of PSD u
        S2[i].append(np.sum(perio ** 2))
        SK[i].append(((M + 1) / (M - 1)) * ((M * S2[i][j] / S1[i][j] ** 2) - 1))

MSK = [[], [], [], [], []]
for i, snr in enumerate(snrs):
    for j, s1 in enumerate(S1[i][:msklen]):
        s1 = 0.25*(S1[i][j] + S1[i][j+1] + S1[i][j+2] + S1[i][j+3])
        s2 = 0.25*(S2[i][j] + S2[i][j+1] + S2[i][j+2] + S2[i][j+3])
        MSK[i].append(((M + 1) / (M - 1)) * ((M * s2 / s1 ** 2) - 1))

plt.figure(0)#, figsize=[22,16])
plt.plot(duty_cycles, SK[0], label="SNR=1")
plt.plot(duty_cycles, SK[1], label="SNR=2")
plt.plot(duty_cycles, SK[2], label="SNR=5")
plt.plot(duty_cycles, SK[3], label="SNR=10")
plt.plot(duty_cycles, SK[4], label="SNR=50")
plt.axhline(y=0.77511, xmin=0,xmax=100, linestyle='--')
plt.axhline(y=1.3254, xmin=0, xmax=100, linestyle='--')
plt.grid()
plt.ylabel("Spectral Kurtosis")
plt.ylim([-1,5])
plt.xlim([0,95])
plt.xlabel("Duty Cycle %")
plt.legend()
plt.savefig('/home/vereese/Documents/PhD/jai-2e/duty.eps', bbox_inches='tight')

"""plt.figure(1)#, figsize=[22,16])
plt.plot(duty_cycles[:msklen], MSK[0], "o", label="SNR=1", linewidth=2)
plt.plot(duty_cycles[:msklen], MSK[1], "o", label="SNR=2", linewidth=2)
plt.plot(duty_cycles[:msklen], MSK[2], "o", label="SNR=5", linewidth=2)
plt.plot(duty_cycles[:msklen], MSK[3], "o", label="SNR=10", linewidth=2)
plt.plot(duty_cycles[:msklen], MSK[4], "o", label="SNR=50", linewidth=2)
plt.axhline(y=0.77511, xmin=0,xmax=100, linestyle='--')
plt.axhline(y=1.3254, xmin=0, xmax=100, linestyle='--')
plt.grid()
plt.ylabel("Spectral Kurtosis")
plt.ylim([-1,4])
plt.xlim([0,95])
plt.xlabel("Duty Cycle %")
plt.legend()
plt.savefig('/home/vereese/Documents/PhD/jai-2e/duty.eps', bbox_inches='tight')
"""

plt.show()