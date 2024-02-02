import numpy as np
from scipy.stats import norm, skewnorm, bernoulli, expon
import matplotlib.pyplot as plt
from constants import thesis_font, a4_textwidth, a4_textheight

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

# Skewness
a1 = 4
a2 = -4
mean1, var1, skew1, kurt1 = skewnorm.stats(a1, moments='mvsk')
mean2, var2, skew2, kurt2 = skewnorm.stats(a2, moments='mvsk')

x1 = np.linspace(skewnorm.ppf(0.01, a2), skewnorm.ppf(0.99, a1), 100)
x2 = np.linspace(skewnorm.ppf(0.01, a2), skewnorm.ppf(0.99, a1), 100)

fig, ax = plt.subplots(1, 1)
ax.plot(x1, skewnorm.pdf(x1, a1), label='$\gamma_1$='+str(round(skew1, 2)), linewidth=2)
ax.plot(x2, skewnorm.pdf(x2, a2), label='$\gamma_1$='+str(round(skew2, 2)), linewidth=2)
ax.set_xlim([x2[0], x1[-1]])
ax.legend()
ax.set_ylabel("$f_X(x)$")
ax.set_xlabel("$x$")
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/skewnessdemo.pdf', transparent=True, bbox_inches='tight')

# Kurtosis
mean, var, skew, kurt = norm.stats(moments='mvsk')
mean_e, var_e, skew_e, kurt_e = expon.stats(moments='mvsk')
p = 0.5
mean_b, var_b, skew_b, kurt_b = bernoulli.stats(p, moments='mvsk')

print("Normal  :")
print("mean    : ", mean)
print("var     : ", var)
print("skew    : ", skew)
print("kurtosis: ", kurt)

print("\nExponential:")
print("mean    : ", mean_e)
print("var     : ", var_e)
print("skew    : ", skew_e)
print("kurtosis: ", kurt_e)

print("\nBernoulli:")
print("mean    : ", mean_b)
print("var     : ", var_b)
print("skew    : ", skew_b)
print("kurtosis: ", kurt_b)

x = np.linspace(norm.ppf(0.01, skew), norm.ppf(0.99, skew), 100)
x_b = np.linspace(bernoulli.ppf(0.01, skew_b), bernoulli.ppf(0.99, skew_b))
x_e = np.linspace(expon.ppf(0.01), expon.ppf(0.99), 100)
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(x_e, expon.pdf(x_e), color='green',label='Leptokurtic $\gamma_2$ = ' + str(kurt_e), linewidth=2)
ax1.plot(x, norm.pdf(x, skew), label='Mesokurtic $\gamma_2$ = ' + str(kurt), linewidth=2)
# Plot Bernoulli manually because it is discrete
ax1.stem([0, 1], [0.5, 0.5], 'red', label='Platykurtic $\gamma_2$ = ' + str(kurt_b))
ax1.set_xlim([-2, 2])
ax1.set_ylim([0, 1])
ax1.legend()
ax1.set_ylabel("$f_X(x)$")
ax1.set_xlabel("$x$")
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/kurtosisdemo.pdf', transparent=True, bbox_inches='tight')
plt.show()
