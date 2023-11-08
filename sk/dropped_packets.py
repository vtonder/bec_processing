import numpy as np
from kurtosis import spectral_kurtosis, spectral_kurtosis_cm
from matplotlib import pyplot as plt
from constants import upper_limit, upper_limit7, lower_limit, lower_limit7

# Setup fonts and sizes for publication, based on page dimensions in inches
textwidth =  128.0 / 25.4 #
textheight = 96.0 / 25.4
font_size = 11
plt.rc('font', size=font_size, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=font_size, labelsize=font_size)
plt.rc(('xtick', 'ytick'), labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

M = 512
num_sk = 100000

perc = np.arange(0,0.2,0.01)
# NOTE: Adding 0s to the data raises the SK. Therefore, if you have dropped packets then you'll increase your SK
#
#print("N: ", N, " x.shape: ", x.shape)
print("% 0s:", 100 * np.sum(np.where(perc == 0, True, False)) / (M*num_sk))

#XF = np.fft.fft(x, axis=1)
mean_sk = []
for p in perc:
    x = np.random.randn(num_sk, M) + 1j * np.random.randn(num_sk, M)
    x[:, 0:int(M*p)] = 0
    mean_sk.append(np.mean(spectral_kurtosis_cm(x, M, 1024)))

plt.figure()
plt.plot(perc*100, (np.asarray(mean_sk) - 1)*100, linewidth=2)
plt.xlabel("% dropped packets")
plt.ylabel("% increase in SK")
plt.xlim([perc[0]*100, perc[-1]*100])
plt.savefig('/home/vereese/Documents/PhD/CASPER2023/casper_presentation/drop.eps', bbox_inches='tight')
#plt.axhline(lower_limit7[M], linestyle = '--', linewidth=2, label="4 $\sigma$ thresholds")
#plt.axhline(upper_limit7[M], linestyle = '--', linewidth=2)
plt.show()