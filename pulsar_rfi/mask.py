import numpy as np

# The first and last 50 channels are masked because the receiver only operates from 900 - 1670MHz and not 856 - 1712 MHz
# GSM : 935.4   - 960.05 MHz <=> 95 - 125
# GNSS: 1145.85 - 1300.9 MHz <=> 347 - 532
# GNSS: 1519.35 - 1608.3 MHz <=> 794 - 900

I = np.load("/home/vereese/git/phd_data/intensity_2210.npy")

masked_I = np.zeros(np.shape(I))
masked_sf = 100*np.ones(1024)
masked_I[51:95,:] = I[51:95,:]
masked_sf[51:95] = np.zeros(44)

masked_I[126:347,:] = I[126:347,:]
masked_sf[126:347] = np.zeros(221)

masked_I[533:794,:] = I[533:794,:]
masked_sf[533:794] = np.zeros(261)

masked_I[901:974,:] = I[901:974,:]
masked_sf[901:974] = np.zeros(73)

np.save("masked_intensity_2210", masked_I)
np.save("masked_sf_2210", masked_sf)