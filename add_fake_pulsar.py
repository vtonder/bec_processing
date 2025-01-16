import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import start_indices, time_chunk_size
from matplotlib import pyplot as plt

original_file = "/home/vereese/1604641064_wide_tied_array_channelised_voltage_0x.h5"
dfo = h5py.File(original_file, 'r')
data0 = dfo['Data/bf_raw'][...]

df = h5py.File('1604640000_fake_pulsar_0x.h5', 'r+')
data = df['Data/bf_raw'][...]
pulse_width = 512
num_samples = pulse_width*1024*2
std1 = 20 
std2 = 40 
num_chunks = len(data[512, :, 0]) / time_chunk_size
print("Number of chunks", num_chunks)
t = np.sin(np.arange(0, np.pi, np.pi/pulse_width)).reshape(pulse_width, 1)
#c = t*np.ones([256,2])
a = np.ones([1024, pulse_width, 2])
sinx = a*t
#e = np.random.normal(0, std, size=num_samples).reshape(1024,256,2)*d
plt.figure(0)
plt.plot(data0[512,:,0], label="original")
plt.figure(1)
plt.plot(data[512,:,0], label="added pulsar")
plt.legend()
plt.show()

#for i in np.arange(num_chunks):
#    if i % 2 == 0:
#        b = np.random.binomial(0,0.7,size=num_samples).reshape(1024, pulse_width, 2)
#        f = np.random.normal(0, std2, size=num_samples).reshape(1024, pulse_width, 2)
#        df['Data/bf_raw'][:, int(8000+(i*time_chunk_size)):int((i*time_chunk_size)+8000+pulse_width), :] = df['Data/bf_raw'][:, int(8000+(i*time_chunk_size)):int((i*time_chunk_size)+8000+pulse_width), :] + np.random.normal(0, std1, size=num_samples).reshape(1024,pulse_width,2) + b*f
#
#df.close()
