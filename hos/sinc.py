from matplotlib import pyplot as plt
import numpy as np

textwidth = 9.6 # 128.0 / 25.4 #
textheight = 7 # 96.0 / 25.4 # 7
plt.rc('font', size=12, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=12, labelsize=12)
plt.rc(('xtick', 'ytick'), labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

phi1 = -2*np.pi
phi2 = 2*np.pi
N = 1000
steps = np.pi / N
range1 = np.arange(phi1,phi2, steps)

a = np.sinc(range1)
b = np.sinc(range1/10)

plt.figure()
plt.plot(range1, b, label="sinc(x/10)")
plt.plot(range1, a, label="sinc(x)")
plt.xlabel("x values")
plt.xlim([range1[0], range1[-1]])
plt.legend()
plt.grid()
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/sinc', bbox_inches='tight')
plt.show()
