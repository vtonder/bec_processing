# MeerKAT bec mode observational information
adc_sample_rate = 1712e6
bw = 856e6
no_channels = 1024
freq_resolution = bw / no_channels
time_resolution = 2 * no_channels / adc_sample_rate
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
vela_T = 1.0 / vela_freq  # vela period

vela_samples_T = vela_T / time_resolution  # samples per vela period

# J0742-2822 
J0742_T = 166.762*10**-3
J0742_freq = 1/J0742_T
J0742_dm = 73.78
J0742_samples_T = J0742_T / time_resolution

