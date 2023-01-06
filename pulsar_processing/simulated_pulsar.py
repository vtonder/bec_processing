import numpy as np
from matplotlib import pyplot as plt
import time

#class Pulsar():
#    def __init__(self):
Ts = 81.92e-6 # sampling time in s
Tobs = 0.1 # observation length in s
fc = 1.4e9 # center frequency in Hz
B = 200e6 # bandwidth in Hz
Nch = 400 # number of channels
freq_resolution = B/Nch

fspin = 200 # pulsar frequency in Hz
DM = 300 # pulsar DM in pc / cm ^ 3
width = 0.3 # fractional width of pulse profile
amp = 4 # amplitude in units of noise amplitude

# time axis
t = np.arange(0,Tobs,Ts)
# frequency axis
f = np.arange(-B/2,B/2,B/Nch)+fc #fc - 0.5 * B:B / Nch: fc + 0.5 * B;
#f = f(1:end - 1) # remove end point

# generate signal
# define(t, f) - grid
[tm, fm] = np.meshgrid(t, f)
# calculate DM delays in s
tdm = 4.15*10**(-3) * DM * ((fc / 1e9)**(-2) - (fm / 1e9)**(-2))
# create signal
stf = amp * np.exp((np.cos(2 * np.pi * fspin * (tm + tdm)) - 1) / width)
ntf = np.random.randn(len(tm))
xtf = stf #+ ntf
# show signal figure
plt.figure(0)
#plt.imshow(t, f / 1e9, xtf)
plt.imshow(xtf)
#axis([0 0.1 min(f / 1e9) max(f / 1e9)])
plt.xlabel('time (s)')
#lt.ylabel('frequency (GHz)')
#plt.show()

fine_fft_len = int(4)

new_freq_res = freq_resolution/fine_fft_len
new_time_res = Ts*fine_fft_len
print("new rime res", new_time_res)
new_pulsar_samples_T = (1/fspin) / new_time_res
new_pulsar_int_samples_T = int(round(new_pulsar_samples_T))

dp = len(xtf[0,:])
print(dp)
seg = int(dp/fine_fft_len)
new_num_pulses = int(np.floor(seg / new_pulsar_samples_T))
print(xtf.shape)
print(new_num_pulses)

redisp_summed_profile = np.zeros([int(3*fine_fft_len/2), new_pulsar_int_samples_T])
meas_chs = np.arange(3) + 10
for i, meas_ch in enumerate(meas_chs):
    data = xtf[meas_ch,0:seg*fine_fft_len].reshape(seg, fine_fft_len)
    print("data")
    #print(data)
    fine_pfb = np.fft.fft(data, fine_fft_len)
    print("fine_pfb")
    print(fine_pfb.shape, new_pulsar_int_samples_T)
    fine_pfb = np.transpose(fine_pfb)

    for j in np.arange(new_num_pulses):

        start = int(j*new_pulsar_samples_T)
        end = start + new_pulsar_int_samples_T

        if end >= seg:
            break

        a = fine_pfb[0:int(fine_fft_len / 2), start:end] * np.conj(fine_pfb[0:int(fine_fft_len / 2), start:end])
        print(a.shape)
        redisp_summed_profile[int(i*fine_fft_len/2):int((i+1)*fine_fft_len/2),:] += a.real

    if i == 0:
        num_2_roll = 0
    else:
        freq = f[10]
        f2 = f[10+i]
        delay = 4.15e-3 * DM * ((f2 / 1e9)**(-2) - (freq / 1e9)**(-2))
        num_2_roll = int(np.round(delay/(Ts)))
    print("roll by", num_2_roll)
    redisp_summed_profile[int(i*fine_fft_len/2):int((i+1)*fine_fft_len/2),:] = np.roll(redisp_summed_profile[int(i*fine_fft_len/2):int((i+1)*fine_fft_len/2),:], num_2_roll)

print(redisp_summed_profile.shape)
plt.figure(1)
#plt.imshow(t, f / 1e9, xtf)
plt.autoscale(True)
plt.imshow(redisp_summed_profile.transpose(), aspect='auto')
#axis([0 0.1 min(f / 1e9) max(f / 1e9)])
plt.xlabel('time (s)')
#lt.ylabel('frequency (GHz)')
plt.show()