import h5py
import glob
import numpy as np

file_names = glob.glob('/home/vereese/pulsar_data/*.h5')
non_zero_indices = {}
for file in file_names:
    data = h5py.File(file, 'r')
    first_non_zero = 0 # Start at index 0

    # search across all 3 dimensions of the data
    for comp in np.arange(2):
        for ch in np.arange(1024):
            temp_non_zero = next((i for i, x in enumerate(data['Data/bf_raw'][ch,:,comp]) if x), None)
            if first_non_zero < temp_non_zero:
                first_non_zero = temp_non_zero

    non_zero_indices.update({file:first_non_zero})

print(non_zero_indices)