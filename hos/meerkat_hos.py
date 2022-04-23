import h5py
import numpy as np
from hos import Bispectrum
from constants import *

vela_x = h5py.File('/home/vereese/pulsar_data/1604641569_wide_tied_array_channelised_voltage_0x.h5', 'r')
data = vela_x['Data/bf_raw'][...]
data = data[:,11620864:,:]

frequencies = np.arange(856+(freq_resolution/1e6)/2,1712+(freq_resolution/1e6)/2,freq_resolution/1e6)
reversed_frequencies = list(reversed(frequencies))
rounded_frequencies = [np.round(f) for f in reversed_frequencies]
gps_l1 = 1575.42
gps_l1_ch = rounded_frequencies.index(round(gps_l1))
gps_l2 = 1227.60
gps_l2_ch = rounded_frequencies.index(round(gps_l2))

b = Bispectrum(data[gps_l1_ch, :, 0], fft_size=1024, reshape=True, method='direct')
b.mean_compensation()
b.bispectrum_I = b.direct_bispectrum()
b.plot_bispectrum_I()

