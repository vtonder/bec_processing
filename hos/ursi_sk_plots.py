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
maxi = max(p1.flatten())/8

plt.figure(0, figsize=[22,16])
plt.imshow(p1, aspect='auto', extent=[0, 1, 0, 1024] , origin='lower', vmin=mini, vmax=maxi)
plt.xlabel('pulse phase')
plt.ylabel('frequency channels')
#plt.savefig('/home/vereese/Documents/PhD/URSI2023/paper/p1', bbox_inches='tight')

plt.figure(1, figsize=[22,16])
plt.imshow(p2, aspect='auto', extent=[0, 1, 0, 1024] , origin='lower', vmin=mini, vmax=maxi)
plt.xlabel('pulse phase')
plt.ylabel('frequency channels')
#plt.savefig('/home/vereese/Documents/PhD/URSI2023/paper/p2', bbox_inches='tight')

plt.figure(2, figsize=[22,16])
plt.imshow(p3, aspect='auto', extent=[0, 1, 0, 1024] , origin='lower', vmin=mini, vmax=maxi)
plt.xlabel('pulse phase')
plt.ylabel('frequency channels')
#plt.savefig('/home/vereese/Documents/PhD/URSI2023/paper/p3', bbox_inches='tight')

plt.figure(3, figsize=[22,16])
plt.imshow(p4, aspect='auto', extent=[0, 1, 0, 1024] , origin='lower', vmin=mini, vmax=maxi)
plt.xlabel('pulse phase')
plt.ylabel('frequency channels')
#plt.savefig('/home/vereese/Documents/PhD/URSI2023/paper/p4', bbox_inches='tight')

plt.show()