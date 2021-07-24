# MeerKAT bec mode observational information
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

# Pulsar information
vela_freq = 11.185 #11.18512 #11.195 # 
#vela_freq = 11.184893175250126
#vela_freq = 11.184168797246667
#tot_obs=(vela_samples_T*time_resolution*22*11) 22 vela pulses , 22 is chosen randomly, that gave 11 subintegrations
#deltaT=(time_resolution*17500) # 17500 is a rough estimate from the plot
# vela_true_period = vela_T*(1+delaT/tot_obs)

vela_dm = 67.99 # to be applied to ms and parsec/cm^3
vela_T = 1.0 / vela_freq  # vela period
c = 4.148808e6 # constant for calculating

vela_samples_T = vela_T / time_resolution  # samples per vela period
