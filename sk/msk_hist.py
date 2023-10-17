import numpy as np
from matplotlib import pyplot as plt
from kurtosis import spectral_kurtosis_cm, s1_s2, ms_spectral_kurtosis_cm
from scipy import signal

# Setup fonts and sizes for publication, based on page dimensions in inches
# This is tuned for LaTeX beamer slides
textwidth =  9.6 #128.0 / 25.4 #
textheight = 7 # 96.0 / 25.4
plt.rc('font', size=11, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=11, labelsize=11)
plt.rc(('xtick', 'ytick'), labelsize=11)
plt.rc('legend', fontsize=11)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

M = 512
m = 2
n = 2
FFT_LEN = 1024
mean = 0
std = 1
N = 100*M*FFT_LEN

pdf_M512 = np.load("/home/vereese/git/phd_data/sk_analysis/2210/pdf_M512.npy")
pdf_M2048 = np.load("/home/vereese/git/phd_data/sk_analysis/2210/pdf_M2048.npy")

wgn_re = np.random.normal(mean, std, size=N)
wgn_im = np.random.normal(mean, std, size=N)
x = (wgn_re + 1j*wgn_im).reshape(FFT_LEN, M, 100)

sk = spectral_kurtosis_cm(x, M, FFT_LEN)
s1,s2 = s1_s2(x, FFT_LEN)
msk = ms_spectral_kurtosis_cm(s1,s2,M,m=m,n=n)

xv = np.arange(0,2,0.01)
plt.figure()
plt.hist(sk.flatten(),1000, density=True, log=True, stacked=True, label="SK hist M=512")
plt.plot(xv,pdf_M512, label="SK PDF M=512", linewidth=2)
plt.hist(msk.flatten(),1000, density=True, log=True, stacked=True, label="MSK hist M=512, n=2, m=2")
plt.plot(xv,pdf_M2048, label="SK PDF M=2048", linewidth=2)
plt.legend()
plt.xlim([0.7,1.3])
plt.ylim([10**-1,10**1])
plt.xlabel("SK values")
plt.savefig("/home/vereese/Documents/PhD/jai-2e/msk_hist")
plt.show()
