# MeerKAT bec mode observational information
adc_sample_rate = 1712e6
bw = 856e6
no_channels = 1024
freq_resolution = bw / no_channels
time_resolution = 2 * no_channels / adc_sample_rate

# Pulsar information
vela_freq = 11.1851 # 11.195
vela_dm = 67.99 # to be applied to ms and parsec/cm^3
vela_T = 1.0 / vela_freq  # vela period
c = 4.148808e6 # constant for calculating

vela_samples_T = vela_T / time_resolution  # samples per vela period
