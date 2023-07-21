import numpy as np
import scipy.stats as stats
from scipy.special import gammaln, loggamma
from matplotlib import pyplot as plt
from scipy.integrate import quad
from kurtosis import spectral_kurtosis
from mpmath import hyp2f1, nstr, re, im
import math

font = {'family': 'STIXGeneral',
        'size': 42}
plt.rc('font', **font)


def hypergeo(a,b,c,z):
    res = hyp2f1(a,b,c,z)
    r = re(res)
    i = im(res)
    return complex(r,i)

N = 1
d = 1
M = 16384

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

#result, error = quad(lambda x: (np.exp(A - m*np.log(1 + ((x-l)/a)**2) - v*np.arctan((x-l)/a))), 0.6, 1.8)

print("max occurs at x: ", xmode)
print("m: ", m)
print("v: ", v)
print("l: ", l)
print("a: ", a)

#P1 of CDF
h1 = np.asarray([hypergeo(1, m+1j * v/2, 2 * m, 2 / (1-1j * ((xi - l) / a))) for xi in x])
h11 = np.asarray([hypergeo(1, m-1j * v/2, 2 * m, 2 / (1-1j * ((xi + l) / a))) for xi in x])
p1 = (a / (2*m - 1)) * (1j - ((x - l) / a)) * p4 * h1
p11 = (a / (2*m - 1)) * (1j - ((x + l) / a)) * p4 * h11

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
sigma3 = 3*np.sqrt(4/M) # theoretical 3 sigma lines
pfa = 0.0013499 # probability of a false alarm
pfa2 = 0.00067495
pfa3 = 0.0026998
print("CDF theoretical 3 sigma upper limit: ", 1+sigma3)
print("CDF theoretical 3 sigma lower limit: ", 1-sigma3)

plt.figure(0, figsize=[22,16])
plt.semilogy(x, p4)
plt.grid()
plt.xlim([0.8, 1.2])
plt.ylim([10**-1, 10**1])
plt.ylabel("SK PDF")
plt.xlabel("SK")
#plt.savefig('/home/vereese/Documents/PhD/URSI2023/paper/pdf', bbox_inches='tight')

plt.figure(1, figsize=[22,16])
plt.semilogy(x, cdf)
plt.semilogy(x, cdf2)
plt.axhline(pfa3, color = 'r')
plt.axvline(1+sigma3, color = 'g', linestyle = '--')
plt.axvline(1-sigma3, color = 'g', linestyle = '--')
plt.axvline(0.77511, color = 'g', linestyle = '-')
plt.axvline(1.3254, color = 'g', linestyle = '-')
plt.xlim([0.65, 1.55])
plt.ylim([10**-7, 10**1])
plt.tight_layout()
plt.ylabel("SK CF and CCF")
plt.xlabel("SK")
plt.grid()
#plt.savefig('/home/vereese/Documents/PhD/URSI2023/paper/cdf', bbox_inches='tight')
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

