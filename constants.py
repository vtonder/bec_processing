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

# SK lower and upper limits

lower_limit = {512:0.77511, 519:0.776424, 1024:0.83425, 1038:0.83527, 1557:0.86281, 2048:0.87892}
upper_limit = {512:1.3254, 519:1.32275, 1024:1.21685, 1038:1.2152, 1557:1.17110, 2048:1.1469}

# first non-zero indices for each file. Obtained using the first_nonzero_indices.py script
# The data files have lots of 0s
# The script searches for the largest first non 0 element across all 3 dimension in the file
# Then made it a multiple of the time chunk size
start_indices = {
 '1604641064_wide_tied_array_channelised_voltage_0x.h5': 0,
 '1604641064_wide_tied_array_channelised_voltage_0y.h5': 13631488,
 '1604641234_wide_tied_array_channelised_voltage_0x.h5': 13631488,
 '1604641234_wide_tied_array_channelised_voltage_0y.h5': 46767104,
 '1604641569_wide_tied_array_channelised_voltage_0x.h5': 13631488,
 '1604641569_wide_tied_array_channelised_voltage_0y.h5': 13631488,
 '1604642210_wide_tied_array_channelised_voltage_0x.h5': 13631488,
 '1604642210_wide_tied_array_channelised_voltage_0y.h5': 13631488,
 '1604642762_wide_tied_array_channelised_voltage_0x.h5': 13631488,
 '1604642762_wide_tied_array_channelised_voltage_0y.h5': 13631488,
 '1604643330_wide_tied_array_channelised_voltage_0x.h5': 13631488,
 '1604643330_wide_tied_array_channelised_voltage_0y.h5': 13631488,
 '1604643883_wide_tied_array_channelised_voltage_0x.h5': 13631488,
 '1604643883_wide_tied_array_channelised_voltage_0y.h5': 13631488,
 '1604644511_wide_tied_array_channelised_voltage_0x.h5': 13631488,
 '1604644511_wide_tied_array_channelised_voltage_0y.h5': 13631488
}
for key, i in start_indices.items():
  start_indices[key] = int(round(i/time_chunk_size)*time_chunk_size)

# Pulsar information
# Vela
vela_freq = 11.185084597305504  # unit Hz 11.185053620637202  #11.185031494489326 #11.18500936838522
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

# J0742-2822 
J0742_T = 166.762*10**-3
J0742_freq = 1/J0742_T
J0742_dm = 73.78
J0742_samples_T = J0742_T*10**6 / time_resolution

