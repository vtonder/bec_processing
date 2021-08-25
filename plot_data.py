import numpy as np
from constants import *
from matplotlib import pyplot as plt


summed_profile = np.load('summed_profile_1569.npy')
for i in np.arange(no_channels):
    mean = np.mean(summed_profile[i, :])
    summed_profile[i, :] = summed_profile[i, :] - mean

plt.figure(0)
# plt.autoscale(True)
plt.imshow(summed_profile, aspect='auto', extent=[0, 1, 856, 1712])  # , origin='lower')
# plt.imshow(summed_profile, aspect='auto')
# plt.plot(summed_profile[292,:])
plt.ylabel('Frequency [MHz]')
plt.xlabel('Pulsar phase')
# plt.imshow(np.roll(summed_profile, -30000, axis=1), aspect='auto', extent=[0,1,856,1712]) #interpolation='nearest'
# plt.colorbar()
# plt.clim([-0.5,7000])
# plt.plot(summed_profile[100,:])
plt.show()

vela_sub_int = np.load('sub_integration_true_period_1569.npy')
print(np.shape(vela_sub_int[:, 0:2500]))
print(np.shape(vela_sub_int))

plt.figure(1)
plt.autoscale(True)
plt.ylabel('sub integrations')
plt.xlabel('Pulsar phase')
plt.imshow(vela_sub_int[:, 10000:25000], aspect='auto', origin='lower')
plt.show()