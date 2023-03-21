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
        start = int(i*time_chunk_size)
        stop = int(start + time_chunk_size)
        dt = data[:,start:stop,:]

        # Throw away 55 ch on each side given receiver band is 900-1670 MHz
        #for ch in np.arange(55,969):
        for ch in np.arange(600,800):
            temp_non_zero = next((j for j, x in enumerate(dt[ch, :, :]) if any(x)), 0)
            if start_index < temp_non_zero:
                start_index = temp_non_zero

    print("before time_chunk_size normalisation: ", start_index)
    start_index = (round(start_index/time_chunk_size))*time_chunk_size
    start_indices.update({file_name:start_index})
    print(start_indices)
    print(file_name, " ", time.time()-t1)

with open("start_indices", "w") as f:
    yaml.safe_dump(start_indices, f)
