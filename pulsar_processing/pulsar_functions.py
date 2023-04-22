import numpy as np 
from constants import vela_samples_T, num_ch, frequencies, freq_resolution, time_resolution, dispersion_constant, pulsars
def square_acc(data, count):
    vela_int_samples_T = int(np.floor(vela_samples_T))
    summed_profile = np.zeros([num_ch, vela_int_samples_T])
    num_data_points = data.shape[1]
    for i in np.arange(count):
        start = int(i * vela_samples_T)
        end = int(start + vela_int_samples_T) 
        if end >= num_data_points:
            break

        summed_profile += np.sum(data[:, start:end, :]**2 , axis=2)
    
    return summed_profile

def incoherent_dedisperse(data, pulsar_tag):
    """

    :param data: summed pulsar profile ie re and im has already been combined. shape is 1024 x samples_T
    :param pulsar_tag: last 4 digits of observation code obtained from the file_name
    :return:
    """

    pulsar = pulsars[pulsar_tag]
    dm = pulsar['dm']
    f2 = 1712 - (freq_resolution / 2)
    for i, freq in enumerate(frequencies):
        delay = 10**6*(dispersion_constant * dm * (1/f2**2 - 1/freq**2)) # us
        num_2_roll = int(np.round(delay/time_resolution)) # samples
        data[i, :] = np.roll(data[i, :], num_2_roll)

    return data
