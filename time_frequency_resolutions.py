import numpy as np
from constants import time_resolution, freq_resolution

# This script is used to calculate values tabulated in J0437 results section of thesis

# time resolutions for SK M values
sk_tr = [time_resolution*2**i for i in range(6,14)]
print("SK time resolution:\n", np.round(sk_tr, 2))

msk_fr = [freq_resolution*2**n for n in range(1,5)]
print("\nMSK frequency resolution:\n", np.round(msk_fr, 2))

print("\n median frequency resolution:\n", np.round(freq_resolution*21, 2))

# total number of pulses that went into a J0437 intensity profile, on which median smoothing is done
num_pulses = 45216
print("\n median time resolution:\n", np.round(time_resolution*num_pulses, 2))
