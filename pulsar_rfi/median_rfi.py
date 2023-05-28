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
        I_median[ch, :] = np.median(window)
    diff = np.abs(I - I_median)

    return diff

def rfi_mit(I, diff, stds):
    global int_samples_T
    for phi in np.arange(int_samples_T):
        std = stds[phi]
        for ch in np.arange(num_ch):
            if diff[ch, phi] >= 4 * std:
                I[ch, phi] = 0
    return I

diff = median_smoothing(I, 11)
stds = np.std(I, axis=0)
I = rfi_mit(I, diff, stds)

diff = median_smoothing(I,11)
stds = np.std(I, axis=0)
I = rfi_mit(I, diff, stds)

diff = median_smoothing(I,11)
stds = np.std(I, axis=0)
I = rfi_mit(I, diff, stds)

diff = median_smoothing(I,11)
stds = np.std(I, axis=0)
I = rfi_mit(I, diff, stds)

print("processing time took: ", time.time()-t1)
#mini = np.min(I_median.flatten())
#maxi = np.max(I_median.flatten())
plt.figure(0)
plt.imshow(I, origin="lower", aspect="auto")#,vmin=mini,vmax=maxi)

plt.show()

