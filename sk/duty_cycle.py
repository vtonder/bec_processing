import numpy as np
from matplotlib import pyplot as plt
from constants import thesis_font, a4_textwidth, a4_textheight
from kurtosis import s1_s2, ms_spectral_kurtosis_cm

# Setup fonts and sizes for publication, based on page dimensions in inches
textwidth =  a4_textwidth
textheight = a4_textheight
font_size = thesis_font
# groups are like plt.figure plt.legend etc
plt.rc('font', size=font_size, family='serif')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=font_size, labelsize=font_size)
plt.rc(('xtick', 'ytick'), labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('lines', markersize=5)
# The following should only be used for beamer
# plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
figheight = 0.65 * textwidth
plt.rc('mathtext', fontset='cm')
# to get this working needed to do: sudo apt install cm-super
plt.rc("text", usetex = True)
plt.rc("figure", figsize = (textwidth, figheight))

M = 512
# duty cycle ito %
#duty_cycles = np.array([45,20,95,30,95,5,95,15,75,9,70,20,80,55,60,97,9,70,19,85,65,88,39,75,55,10])
duty_cycles = np.arange(0,110)
msklen = len(duty_cycles) - 2
duty_samples = (duty_cycles / 100) * M
wgn = np.random.normal(0, 1, size=M) + 1j*np.random.normal(0, 1, size=M)
snrs = [1, 2, 5, 10, 50]
print("len duty_cycles: ", len(duty_cycles))
print("duty_samples[-1]: ", duty_samples[-1])
print("len msklen     : ", msklen)
print("wgn shape      : ", wgn.shape)

SK = [[], [], [], [], []]
MSK = [[], [], [], [], []]
S1 = []
S2 = []

# SK iteration
for i, snr in enumerate(snrs):
    S1.append([])
    S2.append([])
    for j, dc in enumerate(duty_samples):
        s = np.zeros([M])
        s[0:int(dc)] = snr
        s = wgn + s
        perio = np.abs(s)**2
        S1[i].append(perio.sum())  # like the mean of PSD u
        S2[i].append(np.sum(perio ** 2))
        SK[i].append(((M + 1) / (M - 1)) * ((M * S2[i][j] / S1[i][j] ** 2) - 1))
        if snr == 2 and j == 0:
            print("S1[2] at dc = 0", S1[i])
            plt.figure(3)
            plt.plot(S1[i],'o')

# TODO: VMSK can't be implemented with the strategy you're following above because there's not enough data
# VMSK iteration
for i, snr in enumerate(snrs):
    for j, dc in enumerate(duty_samples):
        s = np.zeros([M])
        s[0:int(dc)] = snr
        s = wgn + s

    for j, s1 in enumerate(S1[i][:msklen]):
        x1 = 0.5*(float(S1[i][j]) + float(S1[i][j+1])) # + S1[i][j+2] + S1[i][j+3])
        x2 = 0.5*(float(S2[i][j]) + float(S2[i][j+1])) # + S2[i][j+2] + S2[i][j+3])
        MSK[i].append(((M + 1) / (M - 1)) * ((M * x2 / x1 ** 2) - 1))

plt.figure(0)
plt.plot(duty_cycles, SK[4], label="SNR=50", linewidth=2)
plt.plot(duty_cycles, SK[3], label="SNR=10", linewidth=2)
plt.plot(duty_cycles, SK[2], label="SNR=5", linewidth=2)
plt.plot(duty_cycles, SK[1], label="SNR=2", linewidth=2)
plt.plot(duty_cycles, SK[0], label="SNR=1", linewidth=2)
plt.axhline(y=0.77511, xmin=0,xmax=100, linestyle='--', label="$\pm$3$\sigma$ thresholds", linewidth=2)
plt.axhline(y=1.3254, xmin=0, xmax=100, linestyle='--', linewidth=2)
plt.grid()
plt.ylabel("Spectral Kurtosis")
plt.ylim([-1,5])
plt.xlim([0,95])
plt.xlabel("Duty Cycle \%")
plt.legend()
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/duty.pdf', bbox_inches='tight')
print("msk len: ", MSK)
print("dc len: ", len(duty_cycles))

plt.figure(1)
#plt.plot(duty_cycles[:msklen], MSK[0], "o", label="SNR=1", linewidth=2)
#plt.plot(duty_cycles[:msklen], MSK[1], "o", label="SNR=2", linewidth=2)
plt.plot(duty_cycles, SK[2], "o", label="SK", linewidth=2)
plt.plot(duty_cycles[:msklen], MSK[2], "o", label="MSK", linewidth=2)
#plt.plot(duty_cycles[:msklen], MSK[3], "o", label="SNR=10", linewidth=2)
#lt.plot(duty_cycles[:msklen], MSK[4], "o", label="SNR=50", linewidth=2)
plt.axhline(y=0.77511, xmin=0,xmax=100, linestyle='--', label="thresholds")
plt.axhline(y=1.3254, xmin=0, xmax=100, linestyle='--')
plt.grid()
plt.ylabel("Spectral Kurtosis")
plt.ylim([-1,4])
plt.xlim([0,95])
plt.xlabel("Duty Cycle \%")
plt.legend()
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/duty_msk2.eps', bbox_inches='tight')
plt.show()