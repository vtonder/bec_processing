import h5py
import numpy as np
from hos import Bispectrum
import time
import sys
sys.path.append('..')
from constants import h1_ch, gps_l1_ch, gps_l2_ch, gal_e6_ch
from matplotlib import pyplot as plt

ANALYSE = True
PLOT = False

if ANALYSE:
    t1 = time.time()
    vela_y = h5py.File('/net/com08/data6/vereese/1604641569_wide_tied_array_channelised_voltage_0y.h5', 'r')
    #data = vela_y['Data/bf_raw'][...]
    data_re_h1 = np.transpose(vela_y['Data/bf_raw'][h1_ch,13625088:,0])
    data_im_h1 = np.transpose(vela_y['Data/bf_raw'][h1_ch,13625088:,1])
    data_re_l1 = np.transpose(vela_y['Data/bf_raw'][gps_l1_ch,13625088:,0])
    data_im_l1 = np.transpose(vela_y['Data/bf_raw'][gps_l1_ch,13625088:,1])
    data_re_l2 = np.transpose(vela_y['Data/bf_raw'][gps_l2_ch,13625088:,0])
    data_im_l2 = np.transpose(vela_y['Data/bf_raw'][gps_l2_ch,13625088:,1])
    data_re_g6 = np.transpose(vela_y['Data/bf_raw'][gal_e6_ch,13625088:,0])
    data_im_g6 = np.transpose(vela_y['Data/bf_raw'][gal_e6_ch,13625088:,1])
    print("reading in all channels took: {0} s".format(time.time()-t1))

    b_h1 = Bispectrum(data_re_h1+1j*data_im_h1, reshape=True, fft_size=1024, method='direct')
    b_gps_l1 = Bispectrum(data_re_l1+1j*data_im_l1, reshape=True, fft_size=1024, method='direct')
    b_gps_l2 = Bispectrum(data_re_l2+1j*data_im_l2, reshape=True, fft_size=1024, method='direct')
    b_gal_e6 = Bispectrum(data_re_g6+1j*data_im_g6, reshape=True, fft_size=1024, method='direct')
    #b.mean_compensation()
    #b.calc_power_spectrum()
    t1 = time.time()
    b_h1.bispectrum_I = b_h1.direct_bispectrum()
    b_gps_l1.bispectrum_I = b_gps_l1.direct_bispectrum()
    b_gps_l2.bispectrum_I = b_gps_l2.direct_bispectrum()
    b_gal_e6.bispectrum_I = b_gal_e6.direct_bispectrum()
    print("calculating bispectra took: ", time.time()-t1, " s")
    np.save('h1_bispec', b_h1.bispectrum_I)
    np.save('gps_l1_bispec', b_gps_l1.bispectrum_I)
    np.save('gps_l2_bispec', b_gps_l2.bispectrum_I)
    np.save('gal_e6_bispec', b_gal_e6.bispectrum_I)

if PLOT:
    data = {'gps_l1_bispec.npy':[], 'gps_l2_bispec.npy': [], 'gal_e6_bispec.npy': [], 'h1_bispec.npy':[], 'vela_bispec.npy':[]}

    for i,fn in enumerate(data.keys()):
        data[fn] = np.load(fn)
        plt.figure(i)
        plt.imshow(np.abs(data[fn]), aspect='auto', origin='lower')
        plt.title(fn)

    plt.show()

