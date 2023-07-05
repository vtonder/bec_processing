import numpy as np

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
dispersion_constant = 4.148808e3 # constant for calculating dispersion delay as applied to ms TODO double check if this is correct?

# Popular frequencies in MHz
h1 = 1420.4
gps_l1 = 1575.42
gps_l2 = 1227.60
gal_5b = 1207.14
gal_e6 = 1278.75

# center frequencies of each 1024 subbands
frequencies = np.arange(856 + (freq_resolution / 2), 1712 + (freq_resolution / 2), freq_resolution)
h1_ch = np.abs(frequencies-h1).argmin()
gps_l1_ch = np.abs(frequencies-gps_l1).argmin()
gps_l2_ch = np.abs(frequencies-gps_l2).argmin()
gal_5b_ch = np.abs(frequencies-gal_5b).argmin()
gal_e6_ch = np.abs(frequencies-gal_e6).argmin()
dirty_ch = np.abs(frequencies-1210).argmin()
clean_ch = np.abs(frequencies-1450).argmin()
clean_ch2 = 600
dirty_ch2 = 105

# SK lower and upper limits for PFA=0.13499%. Obtained using sk/sk_thresholds.py script
lower_limit = {512:0.77511, 519:0.776424, 1024:0.83425, 1038:0.83527, 1557:0.86281, 2048:0.87892, 4096:0.91233, 8192:0.93694, 10240:0.94336, 16384:0.95494}
upper_limit = {512:1.3254, 519:1.32275, 1024:1.21685, 1038:1.2152, 1557:1.17110, 2048:1.1469, 4096:1.10073, 8192:1.06971, 10240:1.06193, 16384:1.04847}

# SK lower and upper limits for PFA=0.067495%. Obtained using sk/sk_thresholds.py script
lower_limit2 = {512:0.76188, 1024:0.82425, 2048:0.87146, 4096:0.906784, 8192:0.932844, 16384:0.951866}
upper_limit2 = {512:1.35571, 1024:1.23537, 2048:1.158563, 4096:1.108346, 8192:1.07473, 16384:1.051945}

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
vela_freq = 11.185075 # from Alex's admin.txt
#11.185084597305504  # unit Hz 11.185053620637202  #11.185031494489326 #11.18500936838522
#vela_freq = 11.1946499395 #11.184900190310714 #11.18512 #11.18512 #11.195 # 11.1946499395
#vela_freq = 11.184893175250126
#vela_freq = 11.184168797246667
#tot_obs=(vela_samples_T*time_resolution*22*11) 22 vela pulses , 22 is chosen randomly, that gave 11 subintegrations
#deltaT=(time_resolution*17500) # 17500 is a rough estimate from the plot
# vela_true_period = vela_T*(1+delaT/tot_obs)
vela_dm = 67.97 # from ATNF catalog. units: parsec/cm^3
# vela_dm = 68.0247344970703 <-> where is this from?
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
J0437_T = 1.0/J0437_freq
J0437_samples_T = J0437_T*10**6 / time_resolution
J0437_dm = 2.64476
J0437 = {'name':'J0437-4715',
         'freq':J0437_freq,
         'T':J0437_T,
         'samples_T':J0437_samples_T,
         'dm':J0437_dm
         }

# J0536-7543
J0536_T = 1.245861
J0536_freq = 1.0/J0536_T
J0536_samples_T = J0536_T*10**6 / time_resolution
J0536_dm = 18.58
J0536 = {'name':'J0536-7543',
         'freq':J0536_freq,
         'T':J0536_T,
         'samples_T':J0536_samples_T,
         'dm':J0536_dm
        }

# J0737-3039A
J0737_freq = 44.085374 # Pdot=0.000102
J0737_T = 1.0/J0737_freq
J0737_samples_T = J0737_T*10**6 / time_resolution
J0737_dm = 48.92
J0737 = {'name':'J0737-3039A',
         'freq':J0737_freq,
         'T':J0737_T,
         'samples_T':J0737_samples_T,
         'dm':J0737_dm
         }

# J0742-2822 
J0742_T = 166.762*10**-3
J0742_freq = 1/J0742_T
J0742_dm = 73.78
J0742_samples_T = J0742_T*10**6 / time_resolution
J0742 = {'name':'J0742-2822',
         'freq':J0742_freq,
         'T':J0742_T,
         'samples_T':J0742_samples_T, 
         'dm':J0742_dm
        }

# J1644-4559
J1644_freq = 2.19731
J1644_T = 1.0/J1644_freq
J1644_samples_T = J1644_T*10**6 / time_resolution
J1644_dm = 478.8
J1644 = {'name':'J1644-4559',
         'freq':J1644_freq,
         'T':J1644_T,
         'samples_T':J1644_samples_T,
         'dm':J1644_dm
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
