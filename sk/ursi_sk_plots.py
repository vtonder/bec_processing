import numpy as np
from matplotlib import pyplot as plt

font = {'family': 'STIXGeneral',
        'size': 42}
plt.rc('font', **font)

p1 = np.load('./ursi_data/ursi_profile.npy')
p2 = np.load('./ursi_data/ursi_sf_lu.npy')
p3 = np.load('./ursi_data/ursi_sf_l.npy')
p4 = np.load('./ursi_data/ursi_sf_l2.npy')

mini = min(p1.flatten())
maxi = max(p1.flatten())/4

plt.figure(0, figsize=[22,16])
plt.imshow(p1, aspect='auto', extent=[0, 89.40, 0, 1024] , origin='lower', vmin=mini, vmax=maxi)
plt.xlim([26.8, 62.58])
plt.xlabel('time [ms]')
plt.ylabel('frequency channels')
plt.title('(a) Vela Pulse Profile without RFI mitigation')
plt.savefig('/home/vereese/p1', bbox_inches='tight')

plt.figure(1, figsize=[22,16])
plt.imshow(p2, aspect='auto', extent=[0, 89.40, 0, 1024] , origin='lower', cmap='gray_r', vmin=mini, vmax=maxi) #'vmin=mini, vmax=maxi)
plt.xlim([26.8, 62.58])
plt.xlabel('time [ms]')
plt.ylabel('frequency channels')
plt.title('(b) Upper and lower thresholds are applied, M=512')
plt.savefig('/home/vereese/p2', bbox_inches='tight')

plt.figure(2, figsize=[22,16])
plt.imshow(p3, aspect='auto', extent=[0, 89.40, 0, 1024] , origin='lower', cmap='gray_r', vmin=mini, vmax=maxi) # vmin=mini, vmax=maxi)
plt.xlim([26.8, 62.58])
plt.xlabel('time [ms]')
plt.ylabel('frequency channels')
plt.title('(c) Only lower thresholds are applied, M=512')
plt.savefig('/home/vereese/p3', bbox_inches='tight')

plt.figure(3, figsize=[22,16])
plt.imshow(p4, aspect='auto', extent=[0, 89.40, 0, 1024] , origin='lower', cmap='gray_r', vmin=mini, vmax=maxi) #vmin=mini, vmax=maxi)
plt.xlim([26.8, 62.58])
plt.xlabel('time [ms]')
plt.ylabel('frequency channels')
plt.title('(d) Only lower thresholds are applied, M=2048')
plt.savefig('/home/vereese/p4', bbox_inches='tight')

plt.show()
