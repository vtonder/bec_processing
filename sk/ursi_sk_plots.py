import numpy as np
from matplotlib import pyplot as plt

#DIR = "/home/vereese/Documents/PhD/ThesisTemplate/Figures/"
DIR = "/home/vereese/Documents/PhD/URSI2023/"
DIR_IN = "/home/vereese/git/phd_data/sk_analysis/ursi_data/"
textwidth = 128.0 / 25.4 # 9.6 #
textheight = 96.0 / 25.4 #7
plt.rc('font', size=22, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=22, labelsize=22)
plt.rc(('xtick', 'ytick'), labelsize=22)
plt.rc('legend', fontsize=22)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')
#p1 = np.load(DIR_IN+'ursi_profile.npy')
p2 = np.load(DIR_IN+'ursi_sf_lu.npy')
#p3 = np.load(DIR_IN+'ursi_sf_l.npy')
#p4 = np.load(DIR_IN+'ursi_sf_l2.npy')

mini = min(p2.flatten())
maxi = max(p2.flatten())/4

#plt.figure(0)
#plt.imshow(p1, aspect='auto', extent=[0, 89.40, 0, 1024] , origin='lower', vmin=mini, vmax=maxi)
#plt.xlim([26.8, 62.58])
#plt.xlabel('time [ms]')
#plt.ylabel('frequency channels')
#plt.title('Vela Pulse Profile without RFI mitigation')
#plt.savefig(DIR+'p1', bbox_inches='tight')

plt.figure(1)
plt.imshow(p2, aspect='auto', extent=[0, 89.40, 0, 1024] , origin='lower', cmap='gray_r', vmin=mini, vmax=maxi) #'vmin=mini, vmax=maxi)
plt.xlim([26.8, 62.58])
plt.xlabel('time [ms]')
plt.ylabel('frequency channels')
#plt.title('(b) Upper and lower thresholds are applied, M=512')
plt.savefig(DIR+'p2_pres', bbox_inches='tight')

#plt.figure(2)
#plt.imshow(p3, aspect='auto', extent=[0, 89.40, 0, 1024] , origin='lower', cmap='gray_r', vmin=mini, vmax=maxi) # vmin=mini, vmax=maxi)
#plt.xlim([26.8, 62.58])
#plt.xlabel('time [ms]')
#plt.ylabel('frequency channels')
##plt.title('(c) Only lower thresholds are applied, M=512')
#plt.savefig(DIR+'p3_pres', bbox_inches='tight')

#plt.figure(3)
#plt.imshow(p4, aspect='auto', extent=[0, 89.40, 0, 1024] , origin='lower', cmap='gray_r', vmin=mini, vmax=maxi) #vmin=mini, vmax=maxi)
#plt.xlim([26.8, 62.58])
#plt.xlabel('time [ms]')
#plt.ylabel('frequency channels')
##plt.title('(d) Only lower thresholds are applied, M=2048')
#plt.savefig(DIR+'p4_pres', bbox_inches='tight')

plt.show()
