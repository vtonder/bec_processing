import numpy as np

# constants for plotting
thesis_font = 12 # got from ThesisMain.tex
jai_font = 11 # got from ws-jai.cls
# within \begin{document} part of latex document place the following:
# \typeout{The TEXT WIDTH is \the\textwidth}
# \typeout{The TEXT HEIGHT is \the\textheight}
# this will cause the following lines to be printed out in the log files
# Thesis: TEXT WIDTH is 455.0pt
# Thesis: TEXT HEIGHT is 708.0pt
a4_textwidth = 455.0 / 72 # divide the points (pt) by 72 to get inches
a4_textheight = 708.0 / 72
# beamer_textwidth =  128.0 / 25.4 # From Ludwig
# beamer_textheight = 96.0 / 25.4
beamer_textwidth =  342.2953 / 72 # From \typeout{The TEXT WIDTH is \the\textwidth} in beamer doc
beamer_textheight = 260.48 / 72

#JAI: TEXT WIDTH is 505.89pt
#JAI: TEXT HEIGHT is 655.33755pt
jai_textwidth = 505.89 / 72
jai_textheight = 655.33755 / 72

# MeerKAT bec mode observational information
# When look at the h5py data h5_file_obj['Data/timestamps'] data is 2048 apart. this represents samples
# it takes 2048 adc samples to produce one 1024 ch spectra and therefore the bec mode data steps in 2048 sample sizes
# if no channelisation took place the time res would be 1/adc_sample_rate but we're reducing it by a factor of 2048

# Frequencies given in MHz
adc_sample_rate = 1712
bw = 856
num_ch = 1024
freq_resolution = bw / num_ch # MHz
time_resolution = 2 * num_ch / adc_sample_rate # us

# Chunking
# looks like for dedispersion would need to share 14 chunks for reading from different locations
# and for folding we would need ~ 5 chunks. a vela period spans ~ 5 chunks
# the data is chunked freq X time X re_im in the following sizes:
freq_chunk_size = 1024
time_chunk_size = 16384
re_im_chunk_size = 2
#(1284e6-856e6)/freq_resolution - to get frequency bin
dispersion_constant = 4.148808e3 # constant for calculating dispersion delay unit [MHz^2 cm^3 s / pc]

# Popular frequencies in MHz
h1 = 1420.4
gps_l1 = 1575.42
gps_l2 = 1227.60
gal_5b = 1207.14
gal_e6 = 1278.75

# center frequencies of each 1024 subbands - see Bailes 2020
frequencies = np.arange(856, 1712, freq_resolution)
h1_ch = np.abs(frequencies-h1).argmin()
gps_l1_ch = np.abs(frequencies-gps_l1).argmin()
gps_l2_ch = np.abs(frequencies-gps_l2).argmin()
gal_5b_ch = np.abs(frequencies-gal_5b).argmin()
gal_e6_ch = np.abs(frequencies-gal_e6).argmin()
dirty_ch = np.abs(frequencies-1210).argmin()
clean_ch = np.abs(frequencies-1450).argmin()
clean_ch2 = 600
dirty_ch2 = 105
dme_ch = np.asarray([208, 221, 235, 249])
dme_ch = np.append(dme_ch, np.arange(275,285))
dme_ch = np.append(dme_ch, np.asarray([295, 296, 300, 306,307,308, 319, 322, 325, 334, 335, 336, 340]))

lower_limit_0s = {64:1, 128:1, 256:1, 512:1, 1024:1, 2048:1, 4096:1, 8192:1}
upper_limit_0s = {64:1, 128:1, 256:1, 512:1, 1024:1, 2048:1, 4096:1, 8192:1}

lower_limit_0_5s = {64:0.86425, 128:0.90386, 256:0.93236, 512:0.9527, 1024:0.96704, 2048:0.97707, 4096:0.98411, 8192:0.98884}
upper_limit_0_5s = {64:1.08212, 128:1.06545, 256:1.05075, 512:1.03833, 1024:1.02835, 2048:1.02068, 4096:1.01473, 8192:1.01064}

# 1 sigma <=> PFA ~= 100*((1 - 0.6827) / 2) = 15.86 % per side OR ~= 31.7 % when applying both lower and upper thresholds
lower_limit_1s = {64:0.77587, 128:0.83518, 256:0.8801, 512:0.91375, 1024:0.93856, 2048:0.95626, 4096:0.96895, 8192:0.97814}
upper_limit_1s = {64:1.22077, 128:1.16337, 256:1.11934, 512:1.08613, 1024:1.06163, 2048:1.04378, 4096:1.03107, 8192:1.0219}

# 2 sigma
lower_limit_2s = {64:0.62451, 128:0.71304, 256:0.78529, 512:0.84164, 1024:0.88455, 2048:0.91653, 4096:0.93992, 8192:0.95722}
upper_limit_2s = {64:1.59186, 128:1.41071, 256:1.28287, 512:1.1946, 1024:1.13429, 2048:1.0931, 4096:1.0648, 8192:1.04523}

# 2.5 sigma
lower_limit_2_5s = {64:0.5569, 128:0.65673, 256:0.74107, 512:0.80784, 1024:0.85897, 2048:0.8975, 4096:0.92608, 8192:0.947}
upper_limit_2_5s = {64:1.8457, 128:1.57021, 256:1.3819, 512:1.25676, 1024:1.17422, 2048:1.11942, 4096:1.08249, 8192:1.05733}

# 3 sigma, SK lower and upper limits for PFA = 0.267%. Obtained using sk/sk_thresholds.py script
lower_limit_3s = {64:0.4921, 128:0.602, 256:0.698, 512:0.77511, 519:0.776424, 1024:0.83425, 1038:0.83527, 1557:0.86281, 2048:0.87892, 4096:0.91233, 8192:0.93694, 10240:0.94336, 16384:0.95494}
upper_limit_3s = {64:1.85, 128:1.649, 256:1.5, 512:1.3254, 519:1.32275, 1024:1.21685, 1038:1.2152, 1557:1.17110, 2048:1.1469, 4096:1.10073, 8192:1.06971, 10240:1.06193, 16384:1.04847}

# 4sigma, PFA = 0.00633% both ways or ~ 0.0031% when only applying lower or upper thresholds
lower_limit_4s = {64:0.36388, 128:0.492, 256:0.61376, 512:0.71165, 1024:0.78655, 2048:0.843165, 4096:0.88567, 8192:0.917254, 16384:0.94037, 32768:0.95748, 65536:0.96955}
upper_limit_4s = {64:3.11, 128:2.291, 256:1.784, 512:1.48684, 1024:1.3122, 2048:1.206, 4096:1.138913, 8192:1.09502, 16384:1.06563, 32768:1.04558, 65536:1.031847}

# SK lower and upper limits for PFA=0.13499% total, 0.067495% each way. Obtained using sk/sk_thresholds.py script
lower_limit_0_135p = {64:0.4659, 128:0.5796, 256:0.6808, 512:0.76188, 1024:0.82425, 2048:0.87146, 4096:0.906784, 8192:0.932844, 16384:0.951866}
upper_limit_0_135p = {64:1.95, 128:1.7, 256:1.549, 512:1.35571, 1024:1.23537, 2048:1.158563, 4096:1.108346, 8192:1.07473, 16384:1.051945}

# SK lower and upper limits for PFA=0.54% total, 0.27% lower 0.27% upper. Obtained using sk/sk_thresholds.py script
lower_limit_0_54p = {64:0.5201, 128:0.62575, 256:0.71675, 512:0.78922, 1024:0.84497, 2048:0.88699, 4096:0.91823, 8192:0.941213, 16384:0.95803}
upper_limit_0_54p = {64:1.75, 128:1.571, 256:1.45, 512:1.29463, 1024:1.19796, 2048:1.13474, 4096:1.09269, 8192:1.064214, 16384:1.04468}

# SK lower and upper limits for PFA=2% total, 1% lower 1% upper. Obtained using sk/sk_thresholds.py script
lower_limit_2p = {64:0.5799, 128:0.6761, 256:0.7563, 512:0.819412, 1024:0.867764, 2048:0.904044, 4096:0.930797, 8192:0.950373, 16384:0.964839}
upper_limit_2p = {64:1.6, 128:1.482, 256:1.34596, 512:1.234469, 1024:1.160085, 2048:1.110166, 4096:1.053102, 8192:1.053102, 16384:1.037064}

# SK lower and upper limits for PFA=1% total, 0.5% lower 0.5% upper. Obtained using sk/sk_thresholds.py script
lower_limit_1p = {64:0.5469, 128:0.6483, 256:0.7345, 512:0.802819, 1024:0.855226, 2048:0.894684, 4096:0.923978, 8192:0.945496, 16384:0.960966}
upper_limit_1p = {64:1.7, 128:1.531, 256:1.394, 512:1.266733, 1024:1.180529, 2048:1.123473, 4096:1.085193, 8192:1.059247, 16384:1.041324}

# SK lower and upper limits for PFA=10% total, 5% lower 5% upper. Obtained using sk/sk_thresholds.py script
lower_limit_10p = {64:0.6753, 128:0.75457, 256:0.8178, 512:0.8665, 1024:0.9032, 2048:0.902, 4096:0.95, 8192:0.9646, 16384:0.975}
upper_limit_10p = {64:1.415, 128:1.3161, 256:1.22, 512:1.1539, 1024:1.1075, 2048:1.0751, 4096:1.053, 8192:1.0368, 16384:1.0257}

# Used SK histograms and predicted PDF to analyse SK lower and upper limit within clean data channels, see sk/pulsar_sk_thresholds.py
# Compute min and max values, of min and max sk values per clean channel
# Used max and min values between both X-pol and Y-pol sk data sets
lower_limit_skmin = {64:0.17, 128:0.21, 256:0.25, 512:0.6, 1024:0.7, 2048:0.79, 4096:0.85, 8192:0.89}
upper_limit_skmax = {64:14.68, 128:12.31, 256:10.77, 512:8.41, 1024:6.21, 2048:4.98, 4096:5.63, 8192:5.7}

# median of max sk values of all clean channels
upper_limit9 = {64:5, 128:4.12, 256:3.791, 512:3.244, 1024:2.576, 2048:1.991, 4096:1.564, 8192:1.301, 16384:1.159}

# sampled a view clean channels
upper_limit10 = {64:5, 128:4.2, 256:3.5, 512:3.2, 1024:2.5, 2048:2, 4096:1.9, 8192:1.3, 16384:1.2}

# first non-zero indices for each file. Obtained using the first_nonzero_indices.{py,sh} scripts. 
# The output was written to first_nonzero_indices
first_nonzero_indices = {
  '1604641064_wide_tied_array_channelised_voltage_0x.h5': 10306048,
  '1604641064_wide_tied_array_channelised_voltage_0y.h5': 7668736,
  '1604641234_wide_tied_array_channelised_voltage_0x.h5': 13523200,
  '1604641234_wide_tied_array_channelised_voltage_0y.h5': 8161024,
  '1604641569_wide_tied_array_channelised_voltage_0x.h5': 12952064,
  '1604641569_wide_tied_array_channelised_voltage_0y.h5': 13651712,
  '1604642210_wide_tied_array_channelised_voltage_0x.h5': 12290304,
  '1604642210_wide_tied_array_channelised_voltage_0y.h5': 37167104,
  '1604642762_wide_tied_array_channelised_voltage_0x.h5': 6007040,
  '1604642762_wide_tied_array_channelised_voltage_0y.h5': 13642240,
  '1604643330_wide_tied_array_channelised_voltage_0x.h5': 11668992,
  '1604643330_wide_tied_array_channelised_voltage_0y.h5': 34685952,
  '1604643883_wide_tied_array_channelised_voltage_0x.h5': 11906304,
  '1604643883_wide_tied_array_channelised_voltage_0y.h5': 11587840,
  '1604644511_wide_tied_array_channelised_voltage_0x.h5': 10692096,
  '1604644511_wide_tied_array_channelised_voltage_0y.h5': 9924608
}
# make the indices a multiple of the time chunk size
start_indices = {}
for file_name, idx in first_nonzero_indices.items():
  start_indices.update({file_name : int(round(idx/time_chunk_size)*time_chunk_size)})

# obtained from get_xy_offsets script which takes start indices into account
xy_time_offsets = {
  '1604641064_wide_tied_array_channelised_voltage_0x.h5': 0,
  '1604641064_wide_tied_array_channelised_voltage_0y.h5': 10875392,
  '1604641234_wide_tied_array_channelised_voltage_0x.h5': 0,
  '1604641234_wide_tied_array_channelised_voltage_0y.h5': 38493184,
  '1604641569_wide_tied_array_channelised_voltage_0x.h5': 14370048,
  '1604641569_wide_tied_array_channelised_voltage_0y.h5': 0,
  '1604642210_wide_tied_array_channelised_voltage_0x.h5': 13725952,
  '1604642210_wide_tied_array_channelised_voltage_0y.h5': 0,
  '1604642762_wide_tied_array_channelised_voltage_0x.h5': 38462720,
  '1604642762_wide_tied_array_channelised_voltage_0y.h5': 0,
  '1604643330_wide_tied_array_channelised_voltage_0x.h5': 11616000,
  '1604643330_wide_tied_array_channelised_voltage_0y.h5': 0,
  '1604643883_wide_tied_array_channelised_voltage_0x.h5': 0,
  '1604643883_wide_tied_array_channelised_voltage_0y.h5': 12337920,
  '1604644511_wide_tied_array_channelised_voltage_0x.h5': 0,
  '1604644511_wide_tied_array_channelised_voltage_0y.h5': 10736896
}
"""xy_offsets = np.load("xy_offsets.npy", allow_pickle=True)
for fn, si in start_indices.items():
    start_indices[fn] = int(si + xy_offsets[fn])"""

# Pulsar information is obtained from:
# https://www.atnf.csiro.au/people/joh414/glast/database/summary.html
# Vela
vela_freq = 11.185053620637202 # corrected for pulse period as commented below
#tot_obs=(vela_samples_T*time_resolution*22*11) 22 vela pulses , 22 is chosen randomly, that gave 11 subintegrations
#deltaT=(time_resolution*17500) # 17500 is a rough estimate from the plot
#vela_true_period = vela_T*(1+delaT/tot_obs)
vela_dm = 67.97 # from ATNF v1.64 catalog. units: parsec/cm^3
#vela_dm = 68.02473 # did not yield better results from https://pulsars.org.au/fold/meertime/J0835-4510/
vela_T = 1.0 / vela_freq  # vela period unit s
vela_samples_T = vela_T*10**6 / time_resolution  # samples per vela period
J0835 = {'name':'Vela',
         'freq':vela_freq,
         'T':vela_T,
         'samples_T':vela_samples_T,
         'dm':vela_dm
        } 

# J0437-4715
J0437_freq = 173.69148 # this is as measured by Alex and ATNF = 173.6879458121843
J0437_T = 1.0 / J0437_freq
J0437_samples_T = J0437_T*10**6 / time_resolution
J0437_dm = 2.64476
J0437 = {'name':'J0437-4715',
         'freq':J0437_freq,
         'T':J0437_T,
         'samples_T':J0437_samples_T,
         'dm':J0437_dm
         }

# J0536-7543
J0536_freq = 0.80266
J0536_T = 1 / J0536_freq #1.245861
J0536_samples_T = J0536_T*10**6 / time_resolution
J0536_dm = 18.58
J0536 = {'name':'J0536-7543',
         'freq':J0536_freq,
         'T':J0536_T,
         'samples_T':J0536_samples_T,
         'dm':J0536_dm
        }

# J0737-3039A
J0737_freq = 44.054069392744 #44.085374 # # Pdot=0.000102
J0737_T = 1.0 / J0737_freq
J0737_samples_T = J0737_T*10**6 / time_resolution
J0737_dm = 48.92
J0737 = {'name':'J0737-3039A',
         'freq':J0737_freq,
         'T':J0737_T,
         'samples_T':J0737_samples_T,
         'dm':J0737_dm
         }

# J0742-2822
J0742_freq = 5.9965594774
J0742_T = 1.0 / J0742_freq
J0742_dm = 73.728
J0742_samples_T = J0742_T*10**6 / time_resolution
J0742 = {'name':'J0742-2822',
         'freq':J0742_freq,
         'T':J0742_T,
         'samples_T':J0742_samples_T, 
         'dm':J0742_dm
        }

# J1644-4559
J1644_freq = 2.197424522481 # 2.19731 #
J1644_T = 1.0 / J1644_freq
J1644_samples_T = J1644_T*10**6 / time_resolution
J1644_dm = 478.8
J1644 = {'name':'J1644-4559',
         'freq':J1644_freq,
         'T':J1644_T,
         'samples_T':J1644_samples_T,
         'dm':J1644_dm
         }

gsm_freq = 1733.333
gsm_T = 1.0 / gsm_freq
gsm_samples_T = gsm_T*10**6 / time_resolution
gsm_dm = 0 
gsm = {'name':'GSM',
         'freq':gsm_freq,
         'T':gsm_T,
         'samples_T':gsm_samples_T,
         'dm':gsm_dm
         }

# Dictionary to link code to pulsar
pulsars = {'1234':J0835,
           '1569':J0835,
           '2210':J0437,
           '2762':J0536,
           '3330':J0737,
           '3883':J0742,
           '4511':J1644
          }
