import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import start_indices, time_chunk_size

df = h5py.File('fake_pulsar.h5', 'r+')
data = df['Data/bf_raw'][...]
num_samples = 256*1024*2
std = 20*5
num_chunks = len(data)/time_chunk_size

for i in np.arange(num_chunks):
    df['Data/bf_raw'][:, i*time_chunk_size:(i*time_chunk_size)+256, :] = np.random.normal(0, std, size=num_samples).reshape(1024,256,2)

df.close()