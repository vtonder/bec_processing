import numpy as np
import sys
sys.path.append('../')
from constants import *
from matplotlib import pyplot as plt

font = {'family': 'STIXGeneral',
        'size': 42}
plt.rc('font', **font)

pulsar = pulsars['1234']
mid_T = int(pulsar['samples_T']/2) 
dm = pulsar['dm']

profile_x = np.load('./lower_upper_2048/ps_1234_0x.npy')
profile_y = np.load('./lower_upper_2048/ps_1234_0y.npy')

# summed flags lower upper M = 512 
sf_lu_x = np.load('./lower_upper_2048/summed_flags512_1234_0x.npy')
sf_lu_y = np.load('./lower_upper_2048/summed_flags512_1234_0y.npy')

# summed flags lower M = 512
sf_l_x = np.load('./lower_2048/summed_flags512_1234_0x.npy')
sf_l_y = np.load('./lower_2048/summed_flags512_1234_0y.npy')

# summed flags lower M = 2048 
sf_l2_x = np.load('./lower_2048/summed_flags2048_1234_0x.npy')
sf_l2_y = np.load('./lower_2048/summed_flags2048_1234_0y.npy')

all_data = {"px":profile_x, "py":profile_y, "s1x":sf_lu_x, "s1y":sf_lu_y, "s2x":sf_l_x, "s2y":sf_l_y, "s3x":sf_l2_x, "s3y":sf_l2_y}
f2 = 1712 - (freq_resolution / 2)
for i, freq in enumerate(frequencies):
    delay = 10 ** 6 * (dispersion_constant * dm * (1/(f2**2) - 1 / (freq ** 2)))  # us
    num_2_roll = int(np.round(delay / time_resolution))
    for n, ds in all_data.items():
        all_data[n][i,:] = np.roll(all_data[n][i,:], num_2_roll)

mx = all_data['px'][600,:].argmax()
my = all_data['py'][600,:].argmax()

for n,ds in all_data.items() :
    #m=ds[600,:].argmax()
    if 'x' in n:
        m = mx
    else:
        m = my
    all_data[n] = np.roll(all_data[n], int(mid_T- m))

profile = all_data['px']**2 + all_data['py']**2
sf_lu = all_data['s1x']**2 + all_data['s1y']**2
sf_l = all_data['s2x']**2 + all_data['s2y']**2
sf_l2 = all_data['s3x']**2 + all_data['s3y']**2

np.save('ursi_profile', profile)
np.save('ursi_sf_lu', sf_lu)
np.save('ursi_sf_l', sf_l)
np.save('ursi_sf_l2', sf_l2)

plt.figure(1) #, figsize=[22,16])
plt.imshow(profile, aspect='auto', origin='lower', extent=[0, 1, 0, 1024])
plt.xlabel('pulse phase')
plt.ylabel('frequency channel')

plt.figure(2) #, figsize=[22,16])
plt.imshow(sf_lu, aspect='auto', origin='lower', extent=[0, 1, 0, 1024])
plt.xlabel('pulse phase')
plt.ylabel('frequency channel')

plt.figure(3) #, figsize=[22,16])
plt.imshow(sf_l, aspect='auto', origin='lower', extent=[0, 1, 0, 1024])
plt.xlabel('pulse phase')
plt.ylabel('frequency channel')

plt.figure(4) #, figsize=[22,16])
plt.imshow(sf_l2, aspect='auto', origin='lower', extent=[0, 1, 0, 1024])
plt.xlabel('pulse phase')
plt.ylabel('frequency channel')

plt.show()
