import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter as med
from constants import frequencies, a4_textheight, a4_textwidth, thesis_font

font_size = thesis_font
textwidth = a4_textwidth
textheight = a4_textheight
plt.rc('font', size=font_size, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
plt.rc('axes', titlesize=font_size, labelsize=font_size)
plt.rc(('xtick', 'ytick'), labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

dir = "/home/vereese/git/phd_data/mean_analysis/2210/"
fx_name = "var_0x_1024.npy"
fy_name = "var_0y_1024.npy"
vars_x = np.load(dir + fx_name)
vars_y = np.load(dir + fy_name)

med_x = np.median(vars_x, axis=0)
smoothed = np.abs(vars_x - med_x)
mad_x = np.median(np.abs(vars_x - med_x), axis=0)

mf256_x = med(vars_x, 256)
mf256_y = med(vars_y, 256)

#np.save("std_xpol_2210", np.sqrt(mf256_x))
#np.save("std_ypol_2210", np.sqrt(mf256_y))

plt.figure(0)
plt.plot(frequencies, np.sqrt(vars_x[:, 0]), label="$\sigma_X$", linewidth=2)
plt.plot(frequencies, np.sqrt(mf256_x[:, 0]), label="$\sigma_{mf}$", linewidth=2)
plt.xlabel('Frequencies [MHz]')
plt.ylabel('$\sigma$')
plt.xlim([frequencies[0], frequencies[-1]])
plt.ylim([5, 30])
plt.legend()
#plt.savefig('/home/vereese/Documents/PhD/CASPER2023/casper_presentation/std.eps', bbox_inches='tight')
plt.show()

'''plt.figure(0)
plt.plot(np.sqrt(mf256_x[:,0]), label="re x")
plt.plot(np.sqrt(mf256_x[:,1]), label="im x")
plt.plot(np.sqrt(mf256_y[:,0]), label="re y")
plt.plot(np.sqrt(mf256_y[:,1]), label="im y")
plt.legend()

plt.figure(1)
plt.plot(vars_y[:,1], label="variance")
plt.plot(mf[:,1], label="median filter 128")
plt.plot(mf256_y[:, 1], label="median filter 256")
plt.plot(mf3[:,1], label="median filter 512")
plt.legend()
plt.show()'''