import numpy as np
from matplotlib import pyplot as plt
from hos import Bispectrum
import sys
import time
from matplotlib import cm
from matplotlib.ticker import LinearLocator
sys.path.append("../")
from constants import a4_textwidth, a4_textheight, thesis_font

# TODO: sampling time is not sufficient
# should not upconvert, should calculate what's the offset when the upconverted signal was downconverted using meerkat's pfb and bandpass sampling
# there should be a pfb lib in python

#DIRECTORY = '/home/vereese/git/phd_data/gps_hos/input_data/' # local data location
DIRECTORY = '/home/vereese/data/phd_data/gps_hos/input_data/' # ray data location on NFS
PLOT_TIME = True

# Setup fonts and sizes for publication, based on page dimensions in inches
textwidth =  a4_textwidth
textheight = a4_textheight
font_size = thesis_font
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

class SIM_GPS:
    f_l1 = 1575.42 * 10 ** 6  # Hz
    f_l2 = 1227.6 * 10 ** 6  # Hz
    spb = 6.0 / 300.0  # seconds per bit (spb) = 6 seconds per 300 bits

    L1_L2 = {
        "I_CA_D": f_l2,
        "I_PY": f_l2,
        "I_PY_D": f_l2,
        "Q_CA_D": f_l2,
        "Q_CA": f_l2,
        "Q_L2CM_Dc": f_l2
    }

    def __init__(self, files, bits):
        self.gps_data = {}    # baseband data
        self.gps_data_up = {} # upconverted data
        self.bits = bits
        self.obs_time = self.bits * SIM_GPS.spb # observation time in seconds

        for file in files:
            self.gps_data.update({file:[]})
            self.gps_data_up.update({file:[]})


    def populate(self, direc):
        for file_name, data  in self.gps_data.items():
            with open(direc+file_name) as f:
                for line in f.readlines():
                    data.append(float(line))

    def up_convert(self):

        for file_name, data  in self.gps_data.items():
            data_len = len(data)
            t = np.arange(self.obs_time, step=self.obs_time / data_len)

            if file_name[0] == 'I':
                self.gps_data_up[file_name] = data*np.cos(SIM_GPS.f_l2 * t)
            else:
                self.gps_data_up[file_name] = data*np.sin(SIM_GPS.f_l2 * t)



#gps_file_names = ['I_CA_D_4bit.csv', 'I_CA_D_1bit.csv','I_PY_4bit.csv', 'I_PY_1bit.csv', 'I_PY_D_4bit.csv', 'I_PY_D_1bit.csv',
#                  'Q_CA_D_4bit.csv', 'Q_CA_D_1bit.csv','Q_CA_4bit.csv', 'Q_CA_1bit.csv', 'Q_L2CM_Dc_4bit.csv', 'Q_L2CM_Dc_1bit.csv']
#gps_file_names = ['I_CA_D_2bit.csv','I_PY_2bit.csv', 'I_PY_D_2bit.csv',
#                  'Q_CA_D_2bit.csv','Q_CA_2bit.csv', 'Q_L2CM_2bit.csv']

if len(sys.argv) > 1:
    gps_file_names = []
    for i in np.arange(1, len(sys.argv)):
        gps_file_names.append(str(sys.argv[i]))
else:
    #gps_file_names = ['I_Up1_CA_D_1bit.csv', 'I_Up1_PY_1bit.csv', 'I_Up1_PY_D_1bit.csv', 'Q_Up_CA_D_1bit.csv', 'Q_Up_CA_1bit.csv', 'Q_Up_L2CM_Dc_1bit.csv']#,'I_CA_D_1bit.csv', 'I_PY_1bit.csv', 'I_PY_D_1bit.csv', 'Q_CA_D_1bit.csv', 'Q_CA_1bit.csv', 'Q_L2CM_Dc_1bit.csv']
    #gps_file_names = ['I_Up1_CA_D_1bit.csv', 'I_Up1_PY_1bit.csv', 'I_Up1_PY_D_1bit.csv', 'Q_Up10_CA_D_1bit.csv', 'Q_Up10_CA_1bit.csv', 'Q_Up10_L2CM_Dc_1bit.csv','I_CA_D_1bit.csv', 'I_PY_1bit.csv', 'I_PY_D_1bit.csv', 'Q_CA_D_1bit.csv', 'Q_CA_1bit.csv', 'Q_L2CM_Dc_1bit.csv']
    #gps_file_names = ['I_CA_D_1bit.csv', 'I_PY_1bit.csv', 'I_PY_D_1bit.csv', 'Q_CA_D_1bit.csv', 'Q_CA_1bit.csv', 'Q_L2CM_Dc_1bit.csv']
    gps_code_names = ['$C/A \oplus D$', '$P(Y)$', '$P(Y)\oplus D$', '$C/A \oplus D$', '$C/A$', '$L2CM \oplus D_c$']
    gps_file_names = ['I_CA_D_300bit.csv', 'I_PY_300bit.csv', 'I_PY_D_300bit.csv', 'Q_CA_D_300bit.csv', 'Q_CA_300bit.csv', 'Q_L2CM_Dc_300bit.csv']

gps = SIM_GPS(gps_file_names, 1)
gps.populate(DIRECTORY)
#gps.up_convert()

if PLOT_TIME:
    fig, ax = plt.subplots(3,2, sharex=True, sharey=True)
    k = 0
    for i in np.arange(2):
        for j in np.arange(3):
            ax[j,i].plot(gps.gps_data[gps_file_names[k]][0:200])
            ax[j,i].set_ylabel(gps_code_names[k])
            ax[j,i].grid()

            ax[j,i].set_xlim([0,200])
            if k == 0:
                ax[j, i].set_title("In-phase data")
            elif k == 2:
                ax[j, i].set_xlabel("time samples n")
            elif k == 3:
                ax[j, i].set_title("Quadrature-phase data")
            elif k == 5:
                ax[j, i].set_xlabel("time samples n")
            k += 1

    plt.savefig('/home/vereese/thesis_pics/gps_codes_time_300bit.pdf', bbox_inches='tight')
    plt.show()

"""M = 1024
fs_p = 21.518 # MHz sampling rate for P and P(Y) code
fs_ca = 2.1518 # MHz sampling rate for C/A code
freq_res_p = fs_p/M
freq_res_ca = fs_ca/M
freq_p = np.arange(0,fs_p,freq_res_p)
freq_ca = np.arange(0,fs_ca,freq_res_ca)

bispectra = {}
for i, fn in enumerate(gps_file_names):
    data = gps.gps_data[fn]
    #data_up = gps.gps_data_up[fn]
    data_len = len(data)
    print(fn, "has length", data_len)
    # Bispectra analysis

    fft_len = int(M)
    records = int(np.floor(data_len / fft_len))
    M_2 = int(fft_len / 2)
    cum = np.zeros([records, M_2, M_2], dtype='complex_')
    data = np.asarray(data[0:int(fft_len * records)]).reshape(records, fft_len)
    #data_up = np.asarray(data_up[0:int(fft_len * records)]).reshape(records, fft_len)

    b = Bispectrum(data, method='direct')
    #b_up = Bispectrum(data_up, method='direct')
    b.direct_bispectrum()
    #b.calc_full_bispectrum()
    #b_up.direct_bispectrum()
    #print(b.bispectrum_I-b_up.bispectrum_I)
    # -4 to get rid of .csv
    #np.save(DIRECTORY+fn[:-4]+'_I_bispec_base',b.bispectrum_I)
    #np.save(DIRECTORY+fn[:-4]+'_I_bispec_up',b_up.bispectrum_I)
    #b.calc_power_spectrum()
    #b.plot_full_bispectrum(i, name=fn)
    b.plot_bispectrum_I(i, name=fn)
    #bispectra.update({fn:b})
    #b.plot_power_spectrum(i,fn)
    #print(np.shape(b.full_bispec))

plt.show()

# tested if 2 codes are the same
# tested if gps simulator consistently gives out same pattern for same config and it does
#a = np.abs(bispectra['Q_L2CM_Dc_8bit_1412.csv'].bispectrum_I)
#b = np.abs(bispectra['Q_CA_8bit_1412.csv'].bispectrum_I)
#c = list(np.sum(a - b, axis=0))
#if not any(c): print("They're identical")

# TODO PFB coefficients
# import numpy as np
NCHANS = 1024
NTAPS = 8
NFFT = 2 * NCHANS
# # Effective window length incorporating PFB taps
M = NTAPS * NFFT
# # The PFB coefficients are from a windowed sinc function
pfb_window = np.hamming(M) * np.sinc((np.arange(M) - M / 2.) / NFFT)
print("len PFB window", len(pfb_window))
plt.figure()
plt.plot(pfb_window)
plt.grid()
plt.show()"""