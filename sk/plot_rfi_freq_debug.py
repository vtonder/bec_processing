import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
from constants import frequencies

rfi_data = np.load("/home/vereese/git/phd_data/sk_analysis/2210/rfi_freq_offline.npy")

fig, ax = plt.subplots()
#ax.plot(frequencies,rfi_data[0,:], label="1sigma")
ax.plot(rfi_data[0,:], label="1sigma")
#ax.plot(frequencies,rfi_data[1,:], label="4sigma")
ax.plot(rfi_data[1,:], label="4sigma")
#ax.plot(frequencies,rfi_data[2,:], label="pt")
ax.plot(rfi_data[2,:], label="pt")
#ax.plot(frequencies,rfi_data[3,:], label="med")
ax.plot(rfi_data[3,:], label="med")
ax.legend()
ax.grid()
#ax.axvspan(frequencies[0], frequencies[50], color='blue', alpha=0.5)
#ax.axvspan(frequencies[-50], frequencies[-1], color='blue', alpha=0.5)
#ax.axvspan(frequencies[95], frequencies[126], color='blue', alpha=0.5)
plt.show()