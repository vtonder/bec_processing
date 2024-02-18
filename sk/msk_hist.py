import numpy as np
import sys
from matplotlib import pyplot as plt
from kurtosis import spectral_kurtosis_cm, s1_s2, ms_spectral_kurtosis_cm
sys.path.append('../')
from constants import thesis_font, a4_textwidth, a4_textheight, jai_textwidth, jai_textheight, jai_font

# Setup fonts and sizes for publication, based on page dimensions in inches
textwidth = a4_textwidth
textheight = a4_textheight
font_size = thesis_font

#textwidth = jai_textwidth # a4_textwidth
#textheight = jai_textheight # a4_textheight
#font_size = jai_font # thesis_font

# groups are like plt.figure plt.legend etc
plt.rc('font', size=font_size, family='serif')
plt.rc('pdf', fonttype=42)
# plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=font_size, labelsize=font_size)
plt.rc(('xtick', 'ytick'), labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('lines', markersize=5)
# The following should only be used for beamer
# plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
figheight = 0.65 * textwidth
plt.rc('mathtext', fontset='cm')
# to get this working needed to do: sudo apt install cm-super
plt.rc("text", usetex=True)
plt.rc("figure", figsize=(textwidth, figheight))

M = 512
m = 2
n = 2
FFT_LEN = 1024
mean = 0
std = 1
N = 100 * M * FFT_LEN

# These data sets were once off generated using sk_thresholds.py script
pdf_M512 = np.load("/home/vereese/git/phd_data/sk_analysis/pdf_M512.npy")
pdf_M2048 = np.load("/home/vereese/git/phd_data/sk_analysis/pdf_M2048.npy")

wgn_re = np.random.normal(mean, std, size=N)
wgn_im = np.random.normal(mean, std, size=N)
x = (wgn_re + 1j*wgn_im).reshape(FFT_LEN, M, 100)

sk = spectral_kurtosis_cm(x, M, FFT_LEN)
s1,s2 = s1_s2(x, FFT_LEN)
msk = ms_spectral_kurtosis_cm(s1,s2,M,m=m,n=n)

xv = np.arange(0,2,0.01)
plt.figure()
plt.hist(sk.flatten(),1000, density=True, log=True, stacked=True, label="SK hist $M$=512")
plt.plot(xv, pdf_M512, label="SK PDF $M$ = "+str(M), linewidth=2)
plt.hist(msk.flatten(),1000, density=True, log=True, stacked=True, label="MSK hist $M$=512, $n$=2, $m$=2")
plt.plot(xv, pdf_M2048, label="SK PDF $M$ = "+str(M*n*m), linewidth=2)
# loc = 7 <=> center right
plt.legend(loc=7)
plt.xlim([0.7,1.3])
plt.ylim([10**-1,10**1])
plt.xlabel("SK values")
plt.ylabel("log(PDF)")
plt.savefig("/home/vereese/Documents/PhD/ThesisTemplate/Figures/msk_hist.pdf", transparent=True, bbox_inches='tight')
#plt.savefig("/home/vereese/Documents/PhD/jai-2e/msk_hist.pdf", transparent=True, bbox_inches='tight')
plt.show()
