import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
from constants import num_ch, pulsars
import time

t1 = time.time()
I = np.load("/home/vereese/git/phd_data/intensity_2210.npy")/128**4
int_samples_T = I.shape[1]
pulsar = pulsars['2210']

def median_smoothing(I, window_len=21):
    I_median = np.zeros(I.shape)
    wl_2 = int(window_len/2)
    I_median[0:wl_2, :] = I[0:wl_2, :]
    I_median[num_ch-wl_2:, :] = I[num_ch-wl_2:, :]
    for ch in np.arange(wl_2, num_ch - wl_2):
        window = I[ch - wl_2:ch + wl_2+1, :]
        I_median[ch, :] = np.median(window, axis=0)
    diff = np.abs(I - I_median)

    return diff

def rfi_mit(I, diff):
    global int_samples_T
    for phi in np.arange(int_samples_T):
        data = I[:, phi]
        filtered_data = [d for d in data if d != 0]
        std = np.std(filtered_data)
        I[:, phi] = [0 if diff[ch, phi] >= 3 * std else I[ch, phi] for ch in np.arange(num_ch)]

    return I

diff = median_smoothing(I, 21)
for i in np.arange(6):
    I = rfi_mit(I, diff)

np.save("median_smoothed_I6.npy", I)
print("processing time took: ", time.time()-t1)

plt.figure(0)
plt.imshow(I, origin="lower", aspect="auto")
plt.show()

