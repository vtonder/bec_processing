import numpy as np
from matplotlib import pyplot as plt
from constants import frequencies
from pulsar import PulsarIntensity

textwidth = 9.6 #128.0 / 25.4
textheight = 7 #96.0 / 25.4
plt.rc('font', size=22, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=12, labelsize=12)
plt.rc(('xtick', 'ytick'), labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

DIR = "/home/vereese/git/phd_data/sk_analysis/2210/"
TAG = "2210"
DIR2 = "/home/vereese/git/phd_data/mean_analysis/2210/"
var = np.load(DIR2+"var_0x_1024.npy")

num_pulses = 32*1413 # 32 processors each processed 1413 pulses
M = ["512", "1024", "2048", "4096", "8192", "16384"]
sk = {} # includes lower and upper thresholds with PFA = 0.267% (calculated using 0.0013499 in script)
sk_pfa2 = {} # includes lower and upper thresholds with PFA = 2% (calculated using 0.1 in script)

#load data
I = PulsarIntensity(DIR, TAG,"intensity_2210.npy")
masked = PulsarIntensity(DIR, TAG,"masked_intensity_2210.npy", "masked_sf_2210.npy")
for m in M:
    sk.update({m:PulsarIntensity(DIR, TAG, "sk_intensity_M"+m+"_2210.npy", "sk_summed_flags_M"+m+"_2210.npy")})
for m in M:
    sk_pfa2.update({m:PulsarIntensity(DIR, TAG, "sk_intensity_pfa2_M"+m+"_2210.npy", "sk_pfa2_summed_flags_M"+m+"_2210.npy")})
vt = PulsarIntensity(DIR, TAG, "var_threshold_intensity_2210.npy", "vt_summed_flags_2210.npy")
med = PulsarIntensity(DIR, TAG, "median_smoothed_I4_nearest.npy", "sf_median_nearest.npy")

vt.apply_mask()
vt.get_on_pulse()
med.apply_mask()
med.get_on_pulse()
for m, pi in sk.items():
    pi.apply_mask()
    pi.get_on_pulse()
for m, pi in sk_pfa2.items():
    pi.apply_mask()
    pi.get_on_pulse()

for m, pi in sk.items():
    pi.get_on_pulse()
    print("SK M",m)
    print("pulse start: ", pi.pulse_start)
    print("pulse stop : ", pi.pulse_stop)
    print("pulse width: ", pi.pulse_width, "\n")

plt.figure(0)
for m in M:
    plt.plot(sk[m].get_profile(), label="sk "+m)
plt.plot(vt.get_profile(), label="vt")
plt.plot(med.get_profile(), label="med")
plt.plot(I.get_profile(), label="none")
plt.plot(masked.get_profile(), label="masked")
plt.legend()
plt.show()

'''sk_M16384 = np.load(DIR + "sk_intensity_pfa2_M16384_2210.npy")
sk_M16384[0:50,:] = np.zeros(50)
sk_M16384_sf = 100 * (np.float32(np.load(DIR + "sk_pfa2_summed_flags_M16384_2210.npy")).sum(axis=1) / (2 * 4812 * 1413 * 32))
sk_M8192 = np.load(DIR + "sk_intensity_pfa2_M8192_2210.npy").sum(axis=0)
sk_M8192_sf = 100 * (np.float32(np.load(DIR + "sk_pfa2_summed_flags_M8192_2210.npy")).sum(axis=1) / (2 * 4812 * 1413 * 32))
sk_M4096 = np.load(DIR + "sk_intensity_pfa2_M4096_2210.npy").sum(axis=0)
sk_M4096_sf = 100 * (np.float32(np.load(DIR + "sk_pfa2_summed_flags_M4096_2210.npy")).sum(axis=1) / (2 * 4812 * 1413 * 32))
sk_M2048 = np.load(DIR + "sk_intensity_pfa2_M2048_2210.npy").sum(axis=0)
sk_M2048_sf = 100 * (np.float32(np.load(DIR + "sk_pfa2_summed_flags_M2048_2210.npy")).sum(axis=1) / (2 * 4812 * 1413 * 32))
sk_M1024 = np.load(DIR + "sk_intensity_pfa2_M1024_2210.npy").sum(axis=0)
sk_M1024_sf = 100 * (np.float32(np.load(DIR + "sk_pfa2_summed_flags_M1024_2210.npy")).sum(axis=1) / (2 * 4812 * 1413 * 32))
sk_M512 = np.load(DIR + "sk_intensity_pfa2_M512_2210.npy").sum(axis=0)
sk_M512_sf = 100 * (np.float32(np.load(DIR + "sk_pfa2_summed_flags_M512_2210.npy")).sum(axis=1) / (2 * 4812 * 1413 * 32))


vt = np.load(DIR+).sum(axis=0)
vt_sf = 100 * (np.float32(np.load(DIR+"vt_summed_flags_2210.npy")).sum(axis=1) / (2 * 4812 * 1413 * 32))
med = np.load(DIR+"median_smoothed_I4_2.npy")
median_smoothed = np.load(DIR+"median_smoothed_I4_2.npy").sum(axis=0)
ms_sf = 100*np.load(DIR+"sf_median_2.npy").sum(axis=1)/(4*4812)

#profile peaks and floors for PFA=0.267% none, vt, sk 521 sk 1024, sk 2048, sk 4096, sk 8192, sk 16384, median, mask
#peaks = np.asarray([4.9272, 4.5912, 4.5990,  4.5551, 4.5319, 4.5245, 5.51322, 4.4999, 4.4943, 2.6463])
#floors = np.asarray([3.9735, 3.6779, 3.7504, 3.7117, 3.6814, 3.6602, 3.6422, 3.6277, 3.5239, 2.0502])

#profile peaks and floors for PFA=2% sk 521 sk 1024, sk 2048, sk 4096, sk 8192, sk 16384
peaks = np.asarray([4.5133, 4.4778, 4.4638, 4.381, 4.455, 4.441])
floors = np.asarray([3.7196, 3.6873, 3.6632, 3.6426, 3.6313, 3.618])

FWHM = (peaks - floors)/2 + floors
print(FWHM)
# 1000 brightest pulses
"""sk_b = np.load(DIR+"itegrated_sk_bright_intensity_M2048_2210.npy")
b = np.load(DIR+"intensity_bright_2210.npy")
var_b = np.load(DIR+"var_threshold_bright_intensity_M2048_2210.npy")
p_sk_b = sk_b.sum(axis=0)
p_b = b.sum(axis=0)
p_var_b = var_b.sum(axis=0)

# all pulses except 1000 brightest pulses
sk_nb = np.load(DIR+"itegrated_sk_1000nb_intensity_M2048_2210.npy")
nb = np.load(DIR+"intensity_1000nb_2210.npy")
var_nb = np.load(DIR+"var_threshold_1000nb_intensity_M2048_2210.npy")
p_sk_nb = sk_nb.sum(axis=0)
p_nb = nb.sum(axis=0)
p_var_nb = var_nb.sum(axis=0)

v_nb = []
v_b = []
v_sk_nb = []
v_sk_b = []
v_var_nb = []
v_var_b = []

# summed_flags / by 2 * 4812 * 10 (2 pol * int_samples_T * 1000 pulses) then * 100 -> only / 10
sk_1000nb_sf = np.float32(np.load(DIR+"sk_1000nb_summed_flags_M2048_2210.npy")).sum(axis=1)/(2*4812*10)
sk_bright_sf = np.float32(np.load(DIR+"sk_bright_summed_flags_M2048_2210.npy")).sum(axis=1)/(2*4812*10)
vt_1000nb_sf = np.float32(np.load(DIR+"vt_1000nb_summed_flags_M2048_2210.npy")).sum(axis=1)/(2*4812*10)
vt_bright_sf = np.float32(np.load(DIR+"vt_bright_summed_flags_M2048_2210.npy")).sum(axis=1)/(2*4812*10)
vt_sf = 100*(np.float32(np.load(DIR+"vt_summed_flags_M2048_2210.npy")).sum(axis=1)/(2*4812*1413*32))
median_sf = 100*(np.float32(np.load("../pulsar_rfi/sf_median.npy")).sum(axis=1)/(4*4812))
sk_sf = 100*(np.float32(np.load(DIR+"sk_summed_flags_M2048_2210.npy")).sum(axis=1)/(2*4812*1413*32))

for i in np.arange(1024):
  profile = nb[i,:]/max(nb[i,:])
  v_nb.append(np.sqrt(np.var(profile[0:1000])))

  profile = sk_nb[i,:]/max(sk_nb[i,:])
  v_sk_nb.append(np.sqrt(np.var(profile[0:1000])))

  profile = var_nb[i,:]/max(var_nb[i,:])
  v_var_nb.append(np.sqrt(np.var(profile[0:1000])))

  profile = b[i,:]/max(b[i,:])
  m,c = np.polyfit(np.arange(1000),profile[0:1000],1)
  fit = m*np.arange(1000) + c
  profile2 = profile[0:1000] - fit
  v_b.append(np.sqrt(np.var(profile2)))

  profile = sk_b[i,:]/max(sk_b[i,:])
  m,c = np.polyfit(np.arange(1000),profile[0:1000],1)
  fit = m*np.arange(1000) + c
  profile2 = profile[0:1000] - fit
  v_sk_b.append(np.sqrt(np.var(profile2)))

  profile = var_b[i,:]/max(var_b[i,:])
  m,c = np.polyfit(np.arange(1000),profile[0:1000],1)
  fit = m*np.arange(1000) + c
  profile2 = profile[0:1000] - fit
  v_var_b.append(np.sqrt(np.var(profile2)))

f_ch = 400

plt.figure(0)
plt.plot(p_nb/max(p_nb), label="NB")
plt.plot(p_sk_nb/max(p_sk_nb), label="NB SK")
plt.plot(p_var_nb/max(p_var_nb), label="NB var")

plt.plot(p_b/max(p_b), label="B")
plt.plot(p_var_b/max(p_var_b), label="B var")
plt.plot(p_sk_b/max(p_sk_b), label="B SK")
plt.legend()

plt.figure(1)
plt.plot(v_sk_nb, label="sk NB")
plt.plot(v_nb, label="NB")
plt.plot(v_var_nb, label="var NB")

plt.plot(v_sk_b, label="sk B")
plt.plot(v_b, label="B")
plt.plot(v_var_b, label="var B")
plt.legend()

plt.figure(2)
plt.plot(sk_bright_sf, label="sk 1000 b")
plt.plot(sk_1000nb_sf, label="sk 1000nb")
plt.plot(vt_1000nb_sf, label="vt 1000nb")
plt.plot(vt_bright_sf, label="vt 1000 b")
plt.legend()
plt.title("% RFI flagged")"""

plt.figure(0)
plt.plot(masked_sf, label="mask")
plt.plot( sk_M16384_sf, label="sk 16")
plt.plot( sk_M2048_sf, label="sk 2")
plt.plot( vt_sf, label="vt")
plt.plot( ms_sf, label="med")
plt.legend()
plt.title("% RFI flagged")

plt.figure(1)
plt.plot(sk_M16384/max(sk_M16384), label="sk 16")
plt.plot(sk_M2048/max(sk_M2048), label="sk 2")
plt.plot(vt/max(vt), label="vt")
plt.plot(profile/max(profile), label="none")
plt.plot(masked_profile/max(masked_profile), label="mask")
plt.plot(median_smoothed/max(median_smoothed), label="median")
plt.legend()

plt.figure(2)
plt.plot(sk_M16384, label="sk 16")
plt.plot(sk_M8192, label="sk 8")
plt.plot(sk_M4096, label="sk 4")
plt.plot(sk_M2048, label="sk 2")
plt.plot(sk_M1024, label="sk 1")
plt.plot(sk_M512, label="sk .5")
#plt.plot(vt, label="vt")
#plt.plot(profile, label="none")
#plt.plot(masked_profile, label="mask")
#plt.plot(median_smoothed, label="median")
for i in FWHM:
    plt.axhline(i*10**10)
plt.legend()

plt.figure(3)
plt.plot(frequencies, np.sqrt(var[:,0]), label="real")
plt.plot(frequencies, np.sqrt(var[:,1]), label="imag")
plt.legend()

plt.figure(4)
plt.imshow(med,origin="lower",aspect="auto")

plt.figure(5)
#plt.plot(I.sum(axis=0)/max(I.sum(axis=0)), label="no square")
plt.plot(I[862,:] + 0.5*10**7, label="I ch 862")
plt.plot(med[862,:], label="med I ch 862")
plt.plot(I[600,:] + 0.5*10**7, label="I ch 600")
plt.plot(med[600,:], label="med I ch 600")
plt.legend()
plt.show()'''