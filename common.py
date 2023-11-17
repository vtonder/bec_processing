import numpy as np
from constants import time_chunk_size, num_ch, frequencies

# conduct per frequency channel mean compensation on Intensity data
def mean_compensation(data):
    for i in np.arange(num_ch):
        data[i,:] = data[i,:] - np.mean(data[i,:])

    return data
def non_zero_data(data, std):
    indices = np.where(data == 0, True, False)
    for ch in np.arange(num_ch):
        data[indices[ch,:]] = np.random.normal(0, std[ch], np.sum(indices[ch,:]))

    return data

def get_data_window(start_index, pulse_i, samples_T, int_samples_T, tot_ndp):
    start = start_index + (pulse_i * samples_T)
    end = start + int_samples_T
    chunk_start = int(np.floor(start / time_chunk_size) * time_chunk_size)
    chunk_stop = int(np.ceil(end / time_chunk_size) * time_chunk_size)

    if chunk_stop >= tot_ndp:
        return -1, -1

    return chunk_start, chunk_stop

def get_freq_ch(freq):
    freq_ch = np.abs(frequencies - freq).argmin()

    return freq_ch