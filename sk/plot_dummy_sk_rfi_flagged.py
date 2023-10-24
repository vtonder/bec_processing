import numpy as np
from matplotlib import pyplot as plt

# we expect +- 0.3 % flagged because dummy_sk_integrated_intensity.py was run using both upper and lower thresholds 

num_proc = 32 # num_processors that were used during test
pulses = 5   # each processor processed 5 pulses
num_pol = 2 # summed flags over both x and y polarisation

sf = np.float32(np.load("dummy_sk_sig3_summed_flags_M512_2210.npy"))
rfi_flagged = 100*(sf.sum(axis=1))/(num_pol*num_proc*pulses*sf.shape[1]) 
fig,ax = plt.subplots()
ax.plot(rfi_flagged)
ax.set_xlabel("frequency channels")
ax.set_ylabel("% rfi flagged")
plt.show()
