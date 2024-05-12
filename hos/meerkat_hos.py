import numpy as np
from hos import Bispectrum
import time
import sys
sys.path.append('..')
from constants import h1_ch, gps_l1_ch, gps_l2_ch, gal_e6_ch, beamer_textheight, beamer_textwidth, beamer_font
from matplotlib import pyplot as plt

# Setup fonts and sizes for publication, based on page dimensions in inches
textwidth =  beamer_textwidth
textheight = beamer_textheight
font_size = beamer_font
# groups are like plt.figure plt.legend etc
plt.rc('font', size=font_size, family='serif')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=font_size, labelsize=font_size)
plt.rc(('xtick', 'ytick'), labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('lines', markersize=5)
# The following should only be used for beamer
# plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
figheight = 0.65 * textwidth
plt.rc('mathtext', fontset='cm')
# to get this working needed to do: sudo apt install cm-super
plt.rc("text", usetex = True)
plt.rc("figure", figsize = (textwidth, figheight))

ANALYSE = False
PLOT = True

if ANALYSE:
    import h5py
    t1 = time.time()
    vela_y = h5py.File('/net/com08/data6/vereese/1604641569_wide_tied_array_channelised_voltage_0y.h5', 'r')
    #data = vela_y['Data/bf_raw'][...]
    data_h1 = np.transpose(vela_y['Data/bf_raw'][h1_ch,13625088:,:].astype(np.float))
    data_l1 = np.transpose(vela_y['Data/bf_raw'][gps_l1_ch,13625088:,:].astype(np.float))
    data_l2 = np.transpose(vela_y['Data/bf_raw'][gps_l2_ch,13625088:,:].astype(np.float))
    data_g6 = np.transpose(vela_y['Data/bf_raw'][gal_e6_ch,13625088:,:].astype(np.float))
    print("reading in all channels took: {0} s".format(time.time()-t1))

    b_h1 = Bispectrum(data_h1[:,0]+1j*data_h1[:,1], reshape=True, fft_size=1024, method='direct')
    b_gps_l1 = Bispectrum(data_l1[:,0]+1j*data_l1[:,1], reshape=True, fft_size=1024, method='direct')
    b_gps_l2 = Bispectrum(data_l2[:,0]+1j*data_l2[:,1], reshape=True, fft_size=1024, method='direct')
    b_gal_e6 = Bispectrum(data_g6[:,0]+1j*data_g6[:,1], reshape=True, fft_size=1024, method='direct')
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
    DIR='/home/vereese/git/phd_data/meerkat_hos/'
    clean_data = np.load(DIR + 'clean2_bispec_0y.npy')
    gps_l2_data = np.load(DIR + 'gps_l2_bispec_full.npy')

    plt.figure(0)
    mag1 = np.abs(clean_data)
    maxi1 = np.max(mag1) / 3
    mini1 = np.min(mag1)
    plt.imshow(mag1, aspect='auto', origin='lower', vmin=mini1, vmax=maxi1, extent=[-512, 512, -512, 512])
    plt.colorbar()
    plt.xlabel("frequency samples $k$")
    plt.ylabel("frequency samples $k$")
    plt.savefig("/home/vereese/Documents/PhD/presentation/clean2_bispec_0y.pdf", transparent=True, bbox_inches='tight')

    plt.figure(1)
    mag2 = np.abs(gps_l2_data)
    maxi2 = np.max(mag2) / 7
    mini2 = np.min(mag2)
    plt.imshow(mag2, aspect='auto', origin='lower', vmin=mini2, vmax=maxi2, extent=[-512, 512, -512, 512])
    plt.colorbar()
    plt.xlabel("frequency samples $k$")
    plt.ylabel("frequency samples $k$")
    plt.savefig("/home/vereese/Documents/PhD/presentation/gps_l2_bispec.pdf", transparent=True, bbox_inches='tight')

    """data = {'gps_l1_bispec_full.npy':[], 'gps_l2_bispec_full.npy': [], 'gal_e6_bispec.npy': [], 'gal_5b_bispec_0y.npy': [],
            'h1_bispec_0y.npy':[], 'vela_bispec.npy':[], 'clean_bispec.npy':[], 'dirty_bispec.npy':[], 'clean2_bispec_0y.npy':[], 'dirty2_bispec_0y.npy':[]}
    names = ['gps_l1_bispec_full', 'gps_l2_bispec', 'gal_e6_bispec', 'gal_5b_bispec_0',
            'h1_bispec_0y', 'vela_bispec', 'clean_bispec', 'dirty_bispec', 'clean2_bispec_0y', 'dirty2_bispec_0y']

    for i,fn in enumerate(data.keys()):
        data[fn] = np.load(DIR+fn)
        mag = np.abs(data[fn])
        maxi = np.max(mag)/3
        mini = np.min(mag)
        plt.figure(i)
        plt.imshow(mag, aspect='auto', origin='lower', vmin = mini, vmax = maxi, extent=[-512, 512, -512, 512])
        plt.colorbar()
        plt.xlabel("frequency samples $k$")
        plt.ylabel("frequency samples $k$")
        plt.savefig("/home/vereese/Documents/PhD/presentation/"+names[i]+".pdf", transparent=True, bbox_inches='tight')
        #plt.imshow(np.angle(data[fn]), aspect='auto', origin='lower')
        #plt.title(fn)"""

    plt.show()

