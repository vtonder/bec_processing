import numpy as np
import math
from pulsar_snr import PI, get_profile, get_pho, compute 
from kurtosis import sk_gaus_tran
import sys
sys.path.append('../')
from constants import num_ch

intensity = PI("../", "intensity_z_2210_p45216.npy", "num_nz_z_2210_p45216.npy", initialise=True)
#intensity.compute()

M = 512
M = 4096 

P_M = intensity.samples_T / M # pulsar period / M This is needed to adjust predicted M because not all M windows will contain the on-pulse section 
#for i in np.arange(num_ch): 
i = 600
I_ch = intensity.I[i, :].reshape(1, intensity.samples_T) # channel i's intensity
nz_ch = intensity.nz[i, :].reshape(1, intensity.samples_T) # number of non zero data points that went into summation for channel i 
profile = get_profile(I_ch, nz_ch)
pulse_start, pulse_stop, pulse_width, snr, toa_un = compute(I_ch, nz_ch)
delta = pulse_width / M
bop = math.ceil(delta) # blocks containing on-pulse phase of pulsar. know for M = 512 there is only 1 M block that contains the on pulse out of the ~9.4 M = 512 blocks that's in the pulse period
pho = get_pho(profile)
sk_offset = sk_gaus_tran(pho, delta)
#predicted_sk = ((bop * sk_offset) + ((P_M - bop) * 1)) / P_M
predicted_sk = sk_offset
s1_in = 0.682689492137086 / 2
s1_out = (1-0.682689492137086) / 2

print("SK offset in block that contains op pulse: ", sk_offset)
print("averaged sk: ", predicted_sk)
print("1 sigma inside, only lower: ", s1_in)
print("1 sigma outside, only lower: ", s1_out)

print("mean sk shift ito 1sigma inside: ", ((predicted_sk - 1) / s1_in) * 100)
print("mean sk shift ito 1sigma outside: ", ((predicted_sk - 1) / s1_out) * 100)





