import sys
import h5py
import numpy as np
import glob
from constants import freq_chunk_size, time_chunk_size
import yaml
import time

# The meerkat data has lots of 0s
# This script takes h5py input file
# It calculates the first index to be non 0 per channel
# The ultimate index chosen is the maximum index between all the channels for re and imaginary

file_names = glob.glob('/net/com08/data6/vereese/*.h5')
start_indices = {}
for file_name in file_names:
    t1 = time.time()
    input_data = h5py.File(file_name, 'r')
    data = input_data['Data/bf_raw']
    dl = data.shape[1]
    num_chunk = dl/time_chunk_size
    start_index = 0
    for i in np.arange(num_chunk):
        start = i*time_chunk_size
        stop = start + time_chunk_size
        dt = data[:,start:stop,:]
        for ch in np.arange(freq_chunk_size):
            temp_non_zero = next(i for i, x in enumerate(data[ch, :, :]) if any(x))
            if start_index < temp_non_zero:
                start_index = temp_non_zero

    start_indices.update({file_name:start_index})
    print(start_indices)
    print(file_name, " ", time.time()-t1)

with open("start_indices", "w") as f:
    yaml.safe_dump(start_indices, f)
