import h5py
import numpy as np
from matplotlib import pyplot as plt
from constants import time_resolution, thesis_font, jai_font, start_indices, time_chunk_size

textwidth = 9.6 # inches of page width as per JAI paper for beamer use dimesion: 128.0 / 25.4
textheight = 7 # inches of page height as per JAI paper for beamer use dimesion: 96.0 / 25.4
font_size = thesis_font 
plt.rc('font', size=font_size, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
plt.rc('axes', titlesize=font_size, labelsize=font_size)
plt.rc(('xtick', 'ytick'), labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

fn = '1604641234_wide_tied_array_channelised_voltage_0x.h5'
df = h5py.File('/net/com08/data6/vereese/' + fn, 'r')
start = start_indices[fn] 
end = start + (50 * time_chunk_size) 
dl = end - start # data len 

t = (np.arange(dl) * time_resolution) / 10**3
plt.figure(0)
plt.plot(t, df['Data/bf_raw'][382, start:end, 0], label='ch 382')
plt.plot(t, df['Data/bf_raw'][600, start:end, 0], label='ch 600')
plt.grid()
plt.legend()
plt.xlim([t[0], t[-1]])
plt.xlabel("observation time [ms]")
plt.ylabel("real 8 bit data")
#plt.savefig('/home/vereese/time_series.png', bbox_inches='tight')

plt.show()
