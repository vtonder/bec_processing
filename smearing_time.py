from constants import frequencies, freq_resolution, time_resolution, vela_dm, adc_sample_rate, num_ch
import numpy as np

#TODO: check if things are not out by a factor of 10e6 changed from freq_resolution being in Hz to MHz
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
r = adc_sample_rate/num_ch/nc # Hz
smearing_band = [8.3*10e6 * vela_dm * (freq_resolution/freq**3) for freq in frequencies] # ms
smearing_samples = [int(np.round(smear_time/time_resolution)) for smear_time in smearing_band]

print(smearing_band)
print(smearing_samples)


vela_d = vela_dm/24.10331
vela_ch0_fmin = frequencies[0] - (freq_resolution)/2
vela_ch0_fmax = frequencies[0] + (freq_resolution)/2
td_ch0 = vela_d*((1/(vela_ch0_fmin**2)) - (1/(vela_ch0_fmax**2)))

print("td_ch0", td_ch0*1000)
print("r", r)
print("r*td_ch0", r*td_ch0)