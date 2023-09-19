import numpy as np
from matplotlib import pyplot as plt

textwidth = 9.6 #128.0 / 25.4
textheight = 7 #96.0 / 25.4
plt.rc('font', size=22, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=22, labelsize=22)
plt.rc(('xtick', 'ytick'), labelsize=22)
plt.rc('legend', fontsize=22)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

DIR = "/home/vereese/git/phd_data/sk_analysis/2210/"

# 1000 brightest pulses
sk_b = np.load(DIR+"itegrated_sk_bright_intensity_M2048_2210.npy")
b = np.load(DIR+"itegrated_sk_bright_intensity_2210.npy")
var_b = np.load(DIR+"var_threshold_bright_intensity_M2048_2210.npy")
p_sk_b = sk_b.sum(axis=0)
p_b = b.sum(axis=0)
p_var_b = var_b.sum(axis=0)

# all pulses except 1000 brightest pulses
sk_nb = np.load(DIR+"itegrated_sk_nb_intensity_M2048_2210.npy")
nb = np.load(DIR+"intensity_nb_2210.npy")
var_nb = np.load(DIR+"var_threshold_nb_intensity_M2048_2210.npy")
p_sk_nb = sk_nb.sum(axis=0)
p_nb = nb.sum(axis=0)
p_var_nb = var_nb.sum(axis=0)

plt.figure(0)
plt.plot(p_sk_nb/max(p_sk_nb), label="NB SK")
plt.plot(p_nb/max(p_nb), label="NB")
plt.plot(p_var_nb/max(p_var_nb), label="NB var")
plt.plot(p_sk_b/max(p_sk_b), label="B SK")
plt.plot(p_b/max(p_b), label="B")
plt.plot(p_var_b/max(p_var_b), label="B var")
plt.legend()
plt.title("1000 brightest pulses")
plt.show()