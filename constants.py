# MeerKAT bec mode observational information
# When look at the h5py data h5_file_obj['Data/timestamps'] data is 2048 apart. this represents samples
# it takes 2048 adc samples to produce 1 1024 ch spectra and therefore the bec mode data steps in 2048 sample sizes
# if no channelisation took place the time res would be 1/adc_sample_rate but we're reducing it by a factor of 2048

adc_sample_rate = 1712e6
bw = 856e6
no_channels = 1024
freq_resolution = bw / no_channels
time_resolution = 2 * no_channels / adc_sample_rate

# Chunking
# looks like for dedispersion would need to share 14 chunks for reading from different locations
# and for folding we would need ~ 5 chunks. a vela period spans ~ 5 chunks
# the data is chunked freq X time X re_im in the following sizes:
freq_chunk_size = 1024
time_chunk_size = 16384
re_im_chunk_size = 2
#(1284e6-856e6)/freq_resolution - to get frequency bin
c = 4.148808e6 # constant for calculating dispersion delay

# Pulsar information
# Vela
vela_freq = 11.185084597305504  #11.185053620637202  #11.185031494489326 #11.18500936838522
#vela_freq = 11.1946499395 #11.184900190310714 #11.18512 #11.18512 #11.195 # 
#vela_freq = 11.184893175250126
#vela_freq = 11.184168797246667
#tot_obs=(vela_samples_T*time_resolution*22*11) 22 vela pulses , 22 is chosen randomly, that gave 11 subintegrations
#deltaT=(time_resolution*17500) # 17500 is a rough estimate from the plot
# vela_true_period = vela_T*(1+delaT/tot_obs)

vela_dm = 67.99 # to be applied to ms and parsec/cm^3
# vela_dm = 68.0247344970703 <-> where is this from?
vela_T = 1.0 / vela_freq  # vela period

vela_samples_T = vela_T / time_resolution  # samples per vela period

# J0742-2822 
J0742_T = 166.762*10**-3
J0742_freq = 1/J0742_T
J0742_dm = 73.78
J0742_samples_T = J0742_T / time_resolution

