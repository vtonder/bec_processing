import h5py
import numpy as np
import time
import sys

file_name = sys.argv[1] 
t1 = time.time()
f = h5py.File(file_name, 'r')
print("processing: ", file_name)
data = f['Data/bf_raw'][()]
first_non_zero = 0 # Start at index 0

# search across all 3 dimensions of the data
#for comp in np.arange(2):
for ch in np.arange(55, 969):
    # None is there for special case where all elements are 0
    temp_non_zero = next((i for i, x in enumerate(data[ch,:,:]) if any(x)), 0)
    if first_non_zero < temp_non_zero:
        first_non_zero = temp_non_zero

print(first_non_zero)
print("time:", time.time()-t1)
print()

