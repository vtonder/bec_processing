import numpy as np 
from constants import num_ch, frequencies, freq_resolution, time_resolution, dispersion_constant, pulsars
def square_acc(data, count, pulsar_tag):
    pulsar = pulsars[pulsar_tag]
    samples_T = pulsar['samples_T']
    int_samples_T = int(np.floor(samples_T))
    summed_profile = np.zeros([num_ch, int_samples_T], np.float32)
    num_data_points = data.shape[1]

    for i in np.arange(count):
        start = int(i * int_samples_T)
        end = int(start + int_samples_T)
        if end >= num_data_points:
            break

        summed_profile += np.sum((data[:, start:end, :].astype(np.float32))**2 , axis=2)
    
    return summed_profile

def incoherent_dedisperse(data, pulsar_tag):
    """

    :param data: summed pulsar profile ie re and im has already been combined. shape is num_ch x samples_T
    :param pulsar_tag: last 4 digits of observation code obtained from the file_name
    :return:
    """

    pulsar = pulsars[pulsar_tag]
    dm = pulsar['dm']
    f2 = frequencies[-1]
    for i, freq in enumerate(frequencies):
        delay = 10**6*(dispersion_constant * dm * ((1 / f2**2) - (1 / freq**2))) # us
        num_2_roll = round(delay / time_resolution) # samples
        data[i, :] = np.roll(data[i, :], num_2_roll)

    return data
