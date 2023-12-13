import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
from constants import num_ch, pulsars
import time
import scipy

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

def rfi_mit(I, diff, sf, int_samples_T, num_nz):
    for phi in np.arange(int_samples_T):
        data = I[:, phi]
        filtered_data = [d for d in data if d != 0]
        std = np.std(filtered_data)

        sf[:, phi] = [1 if diff[ch, phi] >= 4 * std else 0 for ch in np.arange(num_ch)]
        I[:, phi] = np.where(diff[:, phi] >= 4 * std, 0, I[:, phi])
        # zero number of non-zero data points that went into I accumulation
        num_nz[:, phi] = np.where(diff[:, phi] >= 4 * std, 0, num_nz[:, phi])

    return I, sf, num_nz

def sk_med():
    M = ["64", "128", "256", "512", "1024", "2048", "4096", "8192"]
    for m in M:
        print(m)
        I = np.load("/home/vereese/git/phd_data/sk_analysis/2210/l4sig/sk_intensity_l4sig_M" + m + "_m1_n1_2210_p45216.npy")
        num_nz = np.load("/home/vereese/git/phd_data/sk_analysis/2210/l4sig/sk_num_nz_l4sig_M" + m + "_m1_n1_2210_p45216.npy")
        int_samples_T = I.shape[1]
        sf = np.zeros(I.shape)
        I_median = scipy.ndimage.median_filter(I,[21,1], mode="wrap")
        diff = np.abs(I - I_median)

        for i in np.arange(4):
            sf_tmp = np.zeros(I.shape)
            I, sf_tmp, num_nz = rfi_mit(I, diff, sf_tmp, int_samples_T, num_nz)
            sf = sf + sf_tmp

        np.save("median_z_r4_sk_l4sig_M" + m + "_2210_p4812.npy", I)
        np.save("nz_median_z_r4_sk_l4sig_M" + m + "_2210_p4812.npy", num_nz)
        np.save("sf_median_z_r4_sk_l4sig_M" + m + "_2210_p4812.npy", sf)

def int_med():
    I = np.load("/home/vereese/git/phd_data/sk_analysis/2210/intensity_z_2210_p45216.npy")
    num_nz = np.load("/home/vereese/git/phd_data/sk_analysis/2210/num_nz_z_2210_p45216.npy")
    int_samples_T = I.shape[1]
    sf = np.zeros(I.shape)
    I_median = scipy.ndimage.median_filter(I, [21, 1], mode="wrap")
    diff = np.abs(I - I_median)

    for i in np.arange(4):
        sf_tmp = np.zeros(I.shape)
        I, sf_tmp, num_nz = rfi_mit(I, diff, sf_tmp, int_samples_T, num_nz)
        sf = sf + sf_tmp

    # g: dropped packets were replaced with Gaussian noise when calculating intensity
    # z: dropped packets were left as 0s calculating intensity
    # ran RFI mitigation 4 times => r4
    np.save("median_z_r4_2210_p4812", I)
    np.save("sf_median_z_2210_p4812", sf)
    np.save("num_nz_median_z_2210_p4812", num_nz)

t1 = time.time()
int_med()
print("processing time took: ", time.time()-t1)

#plt.figure(0)
#plt.imshow(I, origin="lower", aspect="auto")

#plt.figure(1)
#plt.plot((sf.sum(axis=1)*100)/(int_samples_T*4))

#plt.figure(2)
#plt.plot(power_mit.sum(axis=1))

#plt.show()

