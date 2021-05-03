import h5py
import numpy as np
import numba 
import time
from constants import *
from matplotlib import pyplot as plt

vela_x = h5py.File('/home/vereese/pulsar_data/1604641064_wide_tied_array_channelised_voltage_0x.h5', 'r')
vela_y = h5py.File('/home/vereese/pulsar_data/1604641064_wide_tied_array_channelised_voltage_0y.h5', 'r')

summed_profile = np.zeros([1024, int(np.floor(vela_samples_T))])

#@numba.jit(nopython=True)
def square(data):
    return data*data

#@numba.jit(nopython=True)
def add(re, im):
    return square(re) + square(im) 

#@numba.jit(nopython=True)
#def accumulate(summed_data, data, ch):
#    summed_data[ch][:] += data
#    return summed_data

t1 = 0
t2 = 0
diff = 0
for ch in np.arange(1024):
    t1=time.time()
    for i in np.arange(138):
        summed_profile[ch,:] += add(vela_x['Data']['bf_raw'][ch,i*74670:(i+1)*74670,0], vela_x['Data']['bf_raw'][ch,i*74670:(i+1)*74670,1])
    t2=time.time()
    diff = t2-t1
    print('at ch: ', ch, 'took ',  diff, 's')

#summed_profile = sum_data(vela_x['Data']['bf_raw'][:,:,0], vela_x['Data']['bf_raw'][:,:,1])

np.save('summed_profile', summed_profile)      
print('now going to plot')
plt.figure()
plt.imshow(accumulate(summed_profile))
plt.show()
