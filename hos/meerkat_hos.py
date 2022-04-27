import h5py
import numpy as np
from hos import Bispectrum
import time
import sys
sys.path.append('..')
from constants import *
from matplotlib import pyplot as plt

ANALYSE = True
PLOT = False

if ANALYSE:
    vela_y = h5py.File('/home/vereese/pulsar_data/1604641064_wide_tied_array_channelised_voltage_0y.h5', 'r')
    data = vela_y['Data/bf_raw'][...]
    data_re = np.transpose(data[:,11620864:11631104,0])
    data_im = np.transpose(data[:,11620864:11631104,1])

    frequencies = np.arange(856+(freq_resolution/1e6)/2,1712+(freq_resolution/1e6)/2,freq_resolution/1e6)
    reversed_frequencies = list(reversed(frequencies))
    rounded_frequencies = [np.round(f) for f in reversed_frequencies]
    h1 = 1420.4
    gps_l1 = 1575.42
    gps_l1_ch = rounded_frequencies.index(round(gps_l1))
    gps_l2 = 1227.60
    gps_l2_ch = rounded_frequencies.index(round(gps_l2))
    gal_e6 = 1278.75
    gal_e6_ch = rounded_frequencies.index(round(gal_e6))

    b = Bispectrum(data_re+1j*data_im, fft_size=1024, method='direct')
    #b.mean_compensation()
    #b.calc_power_spectrum()
    t1 = time.time()
    b.bispectrum_I = b.direct_bispectrum(compute_fft=False)
    t2 = time.time()
    print("calculating bispectrum took: ", t2-t1, " s")
    np.save('vela_bispec', b.bispectrum_I)

if PLOT:
    data = np.load('vela_bispec_im.npy')
    print(data)
    plt.figure(0)
    plt.imshow(np.abs(data), aspect='auto', origin='lower')
    plt.show()

