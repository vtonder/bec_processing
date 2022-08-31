import numpy as np 
from constants import vela_samples_T, num_ch 
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
