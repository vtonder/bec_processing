import numpy as np
from constants import time_chunk_size, num_ch, frequencies

# conduct per frequency channel mean compensation on Intensity data
def mean_compensation(data):
    for i in np.arange(num_ch):
        data[i,:] = data[i,:] - np.mean(data[i,:])

    return data

def sub_0_noise_per_ch(data, std):
    """
    Substitute 0s in data with Gaussian noise with 0 mean and std standard deviation per frequency channel

    data: 1 chunk of data
    std: an array of 1024 std's per ch
    """
    for ch in np.arange(num_ch):
        indices = np.where(data[ch] == 0, True, False)
        data[ch, indices] = np.random.normal(0, std[ch], np.sum(indices))

    return data

def sub_0_noise(data, std):
    """
    Substitute 0s in data with Gaussian noise with 0 mean and std standard deviation across all frequency channels

    data: 1 chunk of data
    std: scalar 
    """

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

def get_freq_ch(freq):
    freq_ch = np.abs(frequencies - freq).argmin()

    return freq_ch

def get_pulse_window(chunk_start, start_index, pulse_i, samples_T, int_samples_T):
    pulse_start = int(start_index + (pulse_i * samples_T) - chunk_start)
    pulse_stop = pulse_start + int_samples_T

    return pulse_start, pulse_stop

def get_pulse_power(data, pulse_start, pulse_stop):
    pulse = data[:, pulse_start:pulse_stop, :].astype(np.float32)
    sp = np.sum(pulse**2, axis=2)

    return sp

def get_pulse_flags(summed_flags, pulse_start, pulse_stop):
    """
    pf: pulse_flages
    """
    pf = summed_flags[:, pulse_start:pulse_stop].astype(np.float32)

    return pf

