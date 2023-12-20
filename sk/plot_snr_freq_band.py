from sk.pulsar_snr import PI, keep_DME_band
from matplotlib import pyplot as plt

DIR = "/home/vereese/git/phd_data/sk_analysis/2210_floor/l4sig/"

skl = PI(DIR, "sk_intensity_l4sig_M1024_m1_n1_2210_p45216.npy", "sk_num_nz_l4sig_M1024_m1_n1_2210_p45216.npy", "sk_summed_flags_l4sig_M1024_m1_n1_2210_p45216.npy")
dme = PI(DIR, "sk_dme_intensity_l4sigu4sig_M1024_m1_n1_2210_p45216.npy", "sk_dme_num_nz_l4sigu4sig_M1024_m1_n1_2210_p45216.npy", "sk_dme_summed_flags_l4sigu4sig_M1024_m1_n1_2210_p45216.npy")
med = PI(DIR, "median_z_r4_2210_p4812.npy", "num_nz_median_z_2210_p4812.npy" ,"sf_median_z_2210_p4812.npy")

skl.I, skl.nz = keep_DME_band(skl.I, skl.nz)
dme.I, dme.nz = keep_DME_band(dme.I, dme.nz)
med.I, med.nz = keep_DME_band(med.I, med.nz)

print("SK low")
skl.compute()
print("DME")
dme.compute()
print("med")
med.compute()

plt.plot(skl.profile, label = "lower")
plt.plot(dme.profile, label = "DME")
plt.plot(med.profile, label = "med")
plt.legend()
plt.show()