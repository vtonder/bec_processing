import numpy as np
from constants import time_chunk_size, num_ch

# conduct per frequency channel mean compensation on Intensity data
def mean_compensation(data):
    for i in np.arange(num_ch):
        data[i,:] = data[i,:] - np.mean(data[i,:])

    return data
def non_zero_data(data, std):
    indices = np.where(data == 0, True, False)
    data[indices] = np.random.normal(0, std, np.sum(indices))

    return data

def get_data_window(start_index, pulse_i, samples_T, int_samples_T, tot_ndp):
    start = start_index + (pulse_i * samples_T)
    end = start + int_samples_T
    chunk_start = int(np.floor(start / time_chunk_size) * time_chunk_size)
    chunk_stop = int(np.ceil(end / time_chunk_size) * time_chunk_size)

    if chunk_stop >= tot_ndp:
        return -1, -1

    return chunk_start, chunk_stop

# returns SK lower limit based on given key and M value. keys map to those in constants file
def get_low_limit(low_key, M):
    if low_key == 7:
        from constants import lower_limit7 as l
        low_prefix = "l4sig"
    elif low_key == 0:
        from constants import lower_limit as l
        low_prefix = "l3sig"
    else:
        print("LOWER KEY ERROR: Only 0 (3 sigma) and 7 (4 sigma) now supported.")
    
    return l[M], low_prefix

# returns SK upper limit based on given key and M value. keys map to those in constants file
def get_up_limit(up_key, M):
    if up_key == 7:
        from constants import upper_limit7 as u
        up_prefix = "u4sig"
    elif up_key == 0:
        from constants import upper_limit as u
        up_prefix = "u3sig"
    else:
        print("UPPER KEY ERROR: Only 0 (3 sigma) and 7 (4 sigma) now supported.")
   
    return u[M], up_prefix

