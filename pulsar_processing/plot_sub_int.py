import numpy as np
from matplotlib import pyplot as plt

# This script assists in calculating the drift in the pulse period
# It uses eq. 7.5 from the handbook
# sub_int_vela_11.1946499395.npy plots the sub integration over 4 vela pulse phases
# It was produced using the 1604641569_wide_tied_array_channelised_voltage_0x.h5 file
# A sub-integration happens over 22 vela pulses, this number was randomly chosen
# deltaT in samples = 211375
# 11.1946499395*(1-(211375/252755968))

font = {'family': 'STIXGeneral',
        'size': 22}
plt.rc('font', **font)
DIR = '/home/vereese/git/phd_data/pulsar/'
PREFIX = 'sub_int_vela_11.'
files = ['18500936838522.npy', '185031494489326.npy', '185053620637202.npy', '185075.npy', '18512.npy', '1946499395.npy']
vela_sub_int = np.load(DIR+PREFIX+files[0])
last = vela_sub_int[-1,:]
first = vela_sub_int[8,:] #np.roll(np.asarray(vela_sub_int[8,:]),-700)

plt.figure(0)
plt.autoscale(True)
plt.ylabel('observation time [s]')
plt.xlabel('Pulsar phase')
plt.imshow(vela_sub_int, aspect='auto', extent=[0,1,0,300], origin='lower')


plt.figure(1)
plt.plot(last, label='last')
plt.plot(first, label='first')
plt.grid()
plt.legend()

plt.show()