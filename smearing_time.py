from constants import *
import numpy as np

#DM = 2.410331*(10**(-4))*D
DM = 2
D = DM/(2.410331*(10**(-4)))
f0 = 600
bw = 20
fmin = f0 - bw/2
fmax = f0 + bw/2

td_p = D*((1/f0**2) - (1/fmax**2))
td_m = D*((1/fmin**2) - (1/f0**2))

print("td_p:", td_p*10**6)
print("td_m:", td_m*10**6)

# for vela:
nc = 64
r = adc_sample_rate/no_channels/nc # Hz
frequencies = np.arange(856+(freq_resolution/1e6)/2, 1712+(freq_resolution/1e6)/2, freq_resolution/1e6)
smearing_band = [8.3*10e6 * vela_dm * ((freq_resolution/1e6)/freq**3) for freq in frequencies] # ms
smearing_samples = [int(np.round(smear_time/(time_resolution*1000))) for smear_time in smearing_band]

print(smearing_band)
print(smearing_samples)


vela_d = vela_dm/(2.410331*(10**(-4)))
vela_ch0_fmin = frequencies[0] - (freq_resolution/1e6)/2
vela_ch0_fmax = frequencies[0] + (freq_resolution/1e6)/2
td_ch0 = vela_d*((1/(vela_ch0_fmin**2)) - (1/(vela_ch0_fmax**2)))

print("td_ch0", td_ch0*1000)
print("r", r)
print("r*td_ch0", r*td_ch0)