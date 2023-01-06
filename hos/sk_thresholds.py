import numpy as np
import scipy.stats as stats
from scipy.special import gammaln, loggamma
from matplotlib import pyplot as plt
from scipy.integrate import quad
from kurtosis import spectral_kurtosis

alpha = 1 # shaping parameter
beta = 1 # rate parameter

def factorial(n):
    fact = 0
    for i in np.arange(n):
        fact = fact

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
x = np.arange(0.6,1.8,0.01)
A = (- np.log(a) - 0.5 * np.log(np.pi)) + (loggamma(m+1j*(v/2)) + loggamma(m-1j*(v/2)) - loggamma(m-0.5) - loggamma(m))
print("A: ", A)
#p4 = np.exp(A) * (1 + ((x-l)/a)**2)**-m * np.exp(-v*np.arctan((x-l)/a))
p4 = np.exp(A - m*np.log(1 + ((x-l)/a)**2) - v*np.arctan((x-l)/a))

result, error = quad(lambda x: (np.exp(A - m*np.log(1 + ((x-l)/a)**2) - v*np.arctan((x-l)/a))), 0.6, 1.8)

print("max occurs at x: ", xmode)
print("m: ", m)
print("v: ", v)
print("l: ", l)
print("a: ", a)


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

