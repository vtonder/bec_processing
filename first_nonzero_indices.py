import h5py
import glob
import numpy as np
import yaml
import time

file_names = glob.glob('/home/vereese/pulsar_data/*.h5')
non_zero_indices = {}
for file in file_names:
    t1 = time.time()
    h5pyfile = h5py.File(file, 'r')
    print("processing: ", h5pyfile)
    data = h5pyfile['Data/bf_raw'][()]
    first_non_zero = 0 # Start at index 0

    # search across all 3 dimensions of the data
    for comp in np.arange(2):
        for ch in np.arange(1024):
            # None is there for special case where all elements are 0
            temp_non_zero = next((i for i, x in enumerate(data[ch,:,comp]) if x), None)
            if first_non_zero < temp_non_zero:
                first_non_zero = temp_non_zero

    non_zero_indices.update({file:first_non_zero})
    print(non_zero_indices)
    print(file, " ", time.time()-t1)

with open("first_nonzero_indices", "w") as f:
    yaml.safe_dump(non_zero_indices, f)