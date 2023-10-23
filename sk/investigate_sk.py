#
# The distribution of spectral kurtosis (SK).
#
# Ludwig Schwardt
# 20 October 2023
#

import numpy as np
from scipy.special import loggamma
from mpmath import hyp2f1


def spectral_kurtosis(S1, S2, M):
    return (M + 1) / (M - 1) * ((M * S2) / (S1 ** 2) - 1)


def sk_pearson_parameters(M):
    # First four central moments of SK, mu_n
    u1 = 1  # mean
    u2 = (4 * M**2) / ((M-1) * (M+2) * (M+3))  # variance
    # Moments 3 and 4 are in normalised form:
    # beta_1 = mu_3^2 / mu_2^3   (skewness)
    # beta_2 = mu_4 / mu_2^2   (kurtosis)
    b1 = (4 * (M+2) * (M+3) * (5*M-7)**2) / ((M-1) * (M+4)**2 * (M+5)**2)
    b2 = ( (3* (M+2) * (M+3) * (M**3 + 98*M**2 - 185*M + 78))
           / ((M-1) * (M+4) * (M+5) * (M+6) * (M+7)) )

    # Pearson's criterion, kappa
    k = (b1 * (b2+3)**2) / (4 * (4*b2 - 3*b1) * (2*b2 - 3*b1 - 6))

    # Pearson type IV parameters, (m, nu, a, lambda)
    r = (6 * (b2 - b1 - 1)) / (2*b2 - 3*b1 - 6)
    m = (r + 2) / 2
    v = - (r * (r-2) * np.sqrt(b1)) / np.sqrt(16*(r-1) - b1*(r-2)**2)
    # "Statistics of the Spectral Kurtosis Estimator" (Nita & Gary, 2010)
    # had the 16 below as a 6 but that leads to a complex-valued PDF...
    val = u2 * (16*(r-1) - b1*(r-2)**2)
    if val < 0:
        a = 0.25j * np.sqrt(np.abs(val))
    else:
        a = 0.25 * np.sqrt(val)
    l = u1 - 0.25 * (r-2) * np.sqrt(u2*b1)
    return (m, v, a, l, k)


def pearson_iv_pdf(m, v, a, l, x):
    """Pearson type IV pdf."""
    log_c = - np.log(a) - 0.5 * np.log(np.pi) + np.real(
        # loggamma(a+jb) + loggamma(a-jb) is real since the two terms are conjugates
        loggamma(m + 1j*(v/2)) + loggamma(m - 1j*(v/2)) - loggamma(m - 0.5) - loggamma(m)
    )
    return np.exp(log_c - m*np.log(1 + ((x - l)/a)**2) - v*np.arctan((x - l) / a))


def hypergeo(a, b, c, z):
    return complex(hyp2f1(a, b, c, z))


def P1(m, v, a, l, x):
    """Helper function for Pearson IV cdf (Heinrich, 2004)."""
    p4 = pearson_iv_pdf(m, v, a, l, x)
    h1 = hypergeo(1, m + 1j * v/2, 2*m, 2 / (1 - 1j * ((x - l) / a)))
    return (a / (2*m - 1)) * (1j - ((x - l) / a)) * p4 * h1


def P2(m, v, a, l, x):
    """Helper function for Pearson IV cdf (Heinrich, 2004)."""
    term1 = 1 / (1 - np.exp(-np.pi * (v + 2j*m)))
    p4 = pearson_iv_pdf(m, v, a, l, x)
    h2 = hypergeo(1, 2 - 2*m, 2 - m + 1j * v/2, (1 + 1j * ((x - l) / a)) / 2)
    # denom_t2 = np.exp(-np.pi * (v + 2j*m))  # second term of denominator
    # if math.isinf(denom_t2):
    #     term1 = 0
    # else:
    return term1 - ((1j*a / (1j*v - 2*m + 2)) * (1 + ((x - l) / a)**2) * p4 * h2)


def pearson_iv_cdf(m, v, a, l, x):
    if x < l - a * np.sqrt(3):
        P = 1 + P1(m, v, a, l, x)
    elif x <= l + a * np.sqrt(3):
        P = P2(m, v, a, l, x)
    else:
        P = 1 - P1(m, -v, a, -l, -x)
    return np.real(P)


@np.vectorize
def sk_pdf(M, x):
    m, v, a, l, _ = sk_pearson_parameters(M)
    return pearson_iv_pdf(m, v, a, l, x)


@np.vectorize
def sk_cdf(M, x):
    m, v, a, l, _ = sk_pearson_parameters(M)
    return pearson_iv_cdf(m, v, a, l, x)


# xmode = l - (a * v)/(2 * m)
