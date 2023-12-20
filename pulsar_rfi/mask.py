import numpy as np

# The first and last 50 channels are masked because the receiver only operates from 900 - 1670MHz and not 856 - 1712 MHz
# GSM : 935.4   - 960.05 MHz <=> 95 - 126
# GNSS: 1145.85 - 1300.9 MHz <=> 347 - 533
# GNSS: 1519.35 - 1608.3 MHz <=> 794 - 901

I = np.load("/home/vereese/git/phd_data/sk_analysis/2210/intensity_2210.npy")

masked_I = I
masked_sf = 100*np.zeros(1024)
ns = I.shape[1]
masked_I[0:50,:] = np.zeros([50, ns])
masked_sf[0:50] = 100
masked_I[95:126,:] = np.zeros([31, ns])
masked_sf[95:126] = 100
masked_I[347:533,:] = np.zeros([186, ns])
masked_sf[347:533] = 100
masked_I[794:901,:] = np.zeros([107, ns])
masked_sf[794:901] = 100
masked_I[-50:,:] = np.zeros([50, ns])
masked_sf[-50:] = 100

np.save("masked_intensity_2210", masked_I)
np.save("masked_sf_2210", masked_sf)