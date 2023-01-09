import numpy as np
from constants import frequencies, freq_resolution, vela_dm, dispersion_constant, vela_T

smearing_band = [] # ms
for freq in frequencies:
    smearing_band.append(8.3*10e6 * vela_dm * (freq_resolution/freq**3))

print("smearing")
print(smearing_band)

worst_delay = (dispersion_constant*vela_dm*(1/(frequencies[0]**2) - 1/(frequencies[-1]**2)))
num_vela_delay = worst_delay/vela_T

print(worst_delay, num_vela_delay)

# Q1 Is the dispersive delay within a freq ch more at lower / higher frequencies?
# A: dispersive delay is more at lower frequencies
# This also explains the changing of the slope on the frequency vs phase plot
# At higher frequencies the slope is steeper than at lower frequencies
# At lowest frequency bin the delay is 8.4% of vela period
# At highest frequency bin the delay is 1.05% of vela period

# Q2: How much more does the lowest frequency component get delayed as compared with the highest frequency component?
# The worst delay is 288.29 ms ie the lowest freq component arrives 288.29 ms after the highest freq component.
# There'll be 3.23 vela pulses that came during the delay period.