import numpy as np
import scipy.stats as stats
from scipy.special import gammaln, loggamma
from matplotlib import pyplot as plt
from scipy.integrate import quad
from kurtosis import spectral_kurtosis
from mpmath import hyp2f1, nstr, re, im
import math
from investigate_sk import sk_cdf, sk_pdf

#font = {'family': 'STIXGeneral',
#        'size': 42}
#plt.rc('font', **font)

textwidth = 9.6 # 128.0 / 25.4 #
textheight = 7 #96.0 / 25.4 # 7
plt.rc('font', size=12, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=12, labelsize=12)
plt.rc(('xtick', 'ytick'), labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

def hypergeo(a,b,c,z):
    res = hyp2f1(a,b,c,z)
    r = re(res)
    i = im(res)
    return complex(r,i)

N = 1
d = 1
M = 512

u1 = 1
u2 = (4*M**2)/((M-1)*(M+2)*(M+3)) # this is variance
# need to use 3*std for the limits
print("u2", 1+3*np.sqrt(u2), 1-3*np.sqrt(u2), 1 + 3*np.sqrt(4/M))
#u2 = (2*N*d*(N*d + 1)*M**2*gamma(M*N*d + 2))/((M-1)*gamma(M*N*d + 4))
#u3 = ((8*N*d*(N*d + 1)*M**3*gamma(M*N*d + 2))/((M-1)**2*gamma(M*N*d + 6))) * ((N*d+4)*M*N*d - 5*N*d - 2)
#u4 = ((12*N*d*(N*d + 1)*M**4*gamma(M*N*d + 2))/((M-1)**3*gamma(M*N*d + 8))) * ((M**3 * N**4 * d**4) + (3*M**2 *N**4 * d**4) + (M**3 * N**3 * d**3) + (68 * M**2 * N**3 * d**3) - (93 * M * N**3 * N**3) + (125*M**2 * N**2 * d**2) - (245* M* N**2 *d**2) + (84*N**2*d**2) - (32*M*N*d) + (48*N*d) + 24)

b1 = (4*(M+2)*(M+3)*(5*M-7)**2)/((M-1)*(M+4)**2 * (M+5)**2)
b2 = (3*(M+2)*(M+3)*(M**3 + 98*M**2 - 185*M + 78))/((M-1)*(M+4)*(M+5)*(M+6)*(M+7))
print("b1", np.sqrt(b1))
kappa = (b1*(b2+3)**2)/(4*(4*b2-3*b1)*(2*b2-3*b1-6))
print("kappa: ", kappa)
#Pearson type IV
r = (6*(b2-b1-1))/(2*b2 - 3*b1 - 6)
m = (r+2)/2
v = (-r*(r-2)*np.sqrt(b1))/(np.sqrt(16*(r-1) - b1*(r-2)**2))
val = u2*(16*(r-1) - b1*(r-2)**2)
if val < 0:
    a = 1j*0.25*np.sqrt(np.abs(val))
else:
    a = 0.25 * np.sqrt(val)

l = u1 - 0.25*(r-2)*np.sqrt(u2*b1)

xmode = l - (a*v)/(2*m)
x = np.arange(0,2,0.01)
# Pearson type IV PDF
A = (- np.log(a) - 0.5 * np.log(np.pi)) + (loggamma(m+1j*(v/2)) + loggamma(m-1j*(v/2)) - loggamma(m-0.5) - loggamma(m))
p4 = np.exp(A - m*np.log(1 + ((x-l)/a)**2) - v*np.arctan((x-l)/a))
p44 = np.exp(A - m*np.log(1 + ((l-x)/a)**2) + v*np.arctan((l-x)/a))

#result, error = quad(lambda x: (np.exp(A - m*np.log(1 + ((x-l)/a)**2) - v*np.arctan((x-l)/a))), 0.6, 1.8)

print("max occurs at x: ", xmode)
print("m: ", m)
print("v: ", v)
print("l: ", l)
print("a: ", a)

#P1 of CDF
h1 = np.asarray([hypergeo(1, m+1j * v/2, 2 * m, 2 / (1 - 1j*((xi - l) / a))) for xi in x])
h11 = np.asarray([hypergeo(1, m-1j * v/2, 2 * m, 2 / (1 - 1j*((l - xi) / a))) for xi in x])
p1 = (a / (2*m - 1)) * (1j - ((x - l) / a)) * p4 * h1
p11 = (a / (2*m - 1)) * (1j - ((l - x) / a)) * p44 * h11

# P2 of CDF
h2 = np.asarray([hypergeo(1, 2 - 2 * m, 2 - m + (1j * v / 2), (1 + 1j * ((xi - l) / a)) / 2) for xi in x])
denom_t2 = np.exp(-np.pi * (v + 1j * 2 *m)) # second term of denominator
if math.isinf(denom_t2):
    term1 = 0
else:
    term1 = (1/(1 - denom_t2))
p2 = term1 - ((1j * a / (1j * v - 2 * m + 2)) * (1 + ((x - l) / a)**2) * p4 * h2)

cdf = np.zeros(len(x))
x_cdf_low = l - a*np.sqrt(3)
x_cdf_up = l + a*np.sqrt(3)
for i, xi in enumerate(x):
    if xi < x_cdf_low:
        cdf[i] = 1 + p1[i]
    elif xi > x_cdf_up:
        cdf[i] = 1 - p11[i]
    else:
        cdf[i] = p2[i]

cdf2 = 1 - cdf

#lud_pdf = sk_pdf(M, x)
#lud_cdf = sk_cdf(M, x)
#lud_ccdf = 1 - lud_cdf

#print("CHECK CDF: ", np.sum(lud_cdf != cdf))
#print("CHECK PDF: ", np.sum(lud_pdf != p4))

sigma3 = 3*np.sqrt(4/M) # theoretical 3 sigma lines

# probability of false alarm (PFA) according to %%
pfa2p = 0.01 # 2% total PFA, 1% to both sides
pfa0_13p = 0.00067495 # 0.13499% total PFA, 0.067% to both sides
pfa0_54p = 0.0026998 # 0.53996% toal PFA, 0.269% to both sides
pfa1p = 0.005 # 1% PFA,
pfa6p = 0.05 # 10% PFA,

# PFA ito sigma
pfa0_5s = 0.308537538725987     # 0.5 sigma (1-0.382924922548026)/2
pfa1s = 0.15865525393146002     # 1 sigma (1-0.68268949213708)/2
pfa2s = 0.022750131948178987    # (2 sigma) (1-0.954499736103642)/2
pfa2_5s = 0.006209665325775993  # (2.5 sigma) (1-0.987580669348448)/2
pfa3s = 0.0013499               # probability of a false alarm = 0.267% (3sigma)
pfa4s = 3.1671241833008956e-05  # (4 sigma)

print("CDF theoretical 3 sigma upper limit: ", 1+sigma3)
print("CDF theoretical 3 sigma lower limit: ", 1-sigma3)
low_idx = np.abs(pfa1s - cdf).argmin()
up_idx = np.abs(pfa1s - cdf2).argmin()
print("low: ", cdf[low_idx], x[low_idx])
print("up : ", cdf2[up_idx], x[up_idx])

plt.figure(0)
plt.semilogy(x, p4, linewidth=2)
#plt.semilogy(x, lud_pdf, linewidth=2, label = "L")
#plt.plot(p4, linewidth=2)
plt.grid()
plt.xlim([0.8, 1.2])
plt.ylim([10**-1, 10**1])
plt.ylabel("SK PDF")
plt.xlabel("SK")
plt.legend()
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/pdf.eps', bbox_inches='tight')


plt.figure(1) #, figsize=[22,16])
#plt.semilogy(x, lud_cdf, linewidth=2, label = "L")
#plt.semilogy(x, lud_ccdf, linewidth=2, label = "L")
plt.semilogy(x, cdf, linewidth=2)
plt.semilogy(x, cdf2, linewidth=2)
plt.axhline(pfa0_5s, color = 'r', linewidth=2)
plt.axvline(1+sigma3, color = 'g', linewidth=2, linestyle = '--')
plt.axvline(1-sigma3, color = 'g', linewidth=2, linestyle = '--')
plt.axvline(0.77511, color = 'g', linewidth=2, linestyle = '-')
plt.axvline(1.3254, color = 'g', linewidth=2, linestyle = '-')
plt.xlim([0.65, 1.35])
plt.ylim([10**-7, 10**1])
plt.tight_layout()
plt.ylabel("SK CDF and CCDF")
plt.xlabel("SK")
plt.grid()
plt.legend()
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/cdf.eps', bbox_inches='tight')
plt.show()

# Ludwig Code
"""from scipy.special import loggamma as lnG

m = 20.125
v = -32.442
l = 0.652
a = 0.410

x = np.arange(0.6, 1.8, 0.01)
xla = (x - l) / a

ln_p = - np.log(a) - 0.5 * np.log(np.pi) + lnG(m + 0.5j * v) + lnG(m - 0.5j * v) - lnG(m - 0.5) - lnG(m) - m * np.log(1 + xla ** 2) - v * np.arctan(xla)
p = np.exp(ln_p.real)

plt.figure(0)
plt.semilogy(x, p, label='ludwig')
plt.semilogy(x, p4, label='veree')
plt.legend()
plt.grid()"""

