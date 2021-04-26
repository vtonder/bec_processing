import h5py
import numpy as np
from constants import *
from matplotlib import pyplot as plt

vela_x = h5py.File('/home/vereese/pulsar_data/1604641064_wide_tied_array_channelised_voltage_0x.h5', 'r')
vela_y = h5py.File('/home/vereese/pulsar_data/1604641064_wide_tied_array_channelised_voltage_0y.h5', 'r')


summed_profile = np.zeros(int(np.floor(vela_samples_T)))

for i in np.arange(138):
    summed_profile += vela_x['Data']['bf_raw'][800,i:i+74670,0]

plt.figure()
plt.plot(summed_profile)
plt.show()