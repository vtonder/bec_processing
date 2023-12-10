import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
from constants import thesis_font, a4_textwidth, a4_textheight

textwidth = a4_textwidth
textheight = a4_textheight
font_size = thesis_font
plt.rc('font', size=font_size, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=font_size, labelsize=font_size)
plt.rc(('xtick', 'ytick'), labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

fig, ax = plt.subplots(1, 1)
a1 = 4
a2 = -4
mean1, var1, skew1, kurt1 = skewnorm.stats(a1, moments='mvsk')
mean2, var2, skew2, kurt2 = skewnorm.stats(a2, moments='mvsk')

x1 = np.linspace(skewnorm.ppf(0.01, a2), skewnorm.ppf(0.99, a1), 100)
x2 = np.linspace(skewnorm.ppf(0.01, a2), skewnorm.ppf(0.99, a1), 100)
ax.plot(x1, skewnorm.pdf(x1, a1), label='$\gamma_1$='+str(round(skew1, 2)), linewidth=2)
ax.plot(x2, skewnorm.pdf(x2, a2), label='$\gamma_1$='+str(round(skew2, 2)), linewidth=2)
ax.set_xlim([x2[0], x1[-1]])
ax.legend()
ax.set_ylabel("$f_x(x)$")
ax.set_xlabel("$x$")
#plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/skewnessdemo.png', transparent=True, bbox_inches='tight')
plt.show()


#ax.plot(x, skewnorm.pdf(x, a), 'r-', lw=5, alpha=0.6, label='skewnorm pdf')
#rv = skewnorm(a)
#ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
#vals = skewnorm.ppf([0.001, 0.5, 0.999], a)
#np.allclose([0.001, 0.5, 0.999], skewnorm.cdf(vals, a))
#r = skewnorm.rvs(a, size=1000)
#ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)