import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from math import isnan

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ch350    = np.load("/home/vereese/git/phd_data/sk_analysis/1234/ch350_1234x.npy")
ch600    = np.load("/home/vereese/git/phd_data/sk_analysis/1234/ch600_1234x.npy")
sf_ch350 = np.load("/home/vereese/git/phd_data/sk_analysis/1234/lower_upper_pics/sf512_ch350_1234x.npy")
skf_ch350 = np.load("/home/vereese/git/phd_data/sk_analysis/1234/lower_upper_pics/skf512_ch350_1234x.npy")
sf_ch600 = np.load("/home/vereese/git/phd_data/sk_analysis/1234/lower_upper_pics/sf512_ch600_1234x.npy")
skf_ch600 = np.load("/home/vereese/git/phd_data/sk_analysis/1234/lower_upper_pics/skf512_ch600_1234x.npy")
sk_ch350 = np.load("/home/vereese/git/phd_data/sk_analysis/1234/sk_350.npy")
sk_ch600 = np.load("/home/vereese/git/phd_data/sk_analysis/1234/sk_600.npy")
si = 13631488

ups_skf_ch350 = np.array([])
for i in sk_ch350[2500:5000]:
    tmp = np.asarray([i]*512)
    ups_skf_ch350 = np.concatenate((ups_skf_ch350, tmp))

ups_skf_ch600 = np.array([])
for i in sk_ch600[2500:5000]:
    tmp = np.asarray([i]*512)
    ups_skf_ch600 = np.concatenate((ups_skf_ch600, tmp))

#ups_skf_ch600 = np.array([])
#for i in skf_ch600:
#    tmp = np.asarray([i]*512)
#    ups_skf_ch600 = np.concatenate((ups_skf_ch600, tmp))
d_rfi = (ch350[si+(2500*512):si+(5000*512),0]/128) + (ch350[si+(2500*512):si+(5000*512), 1]/128)
d_no_rfi = (ch600[si+(2500*512):si+(5000*512),0]/128) + (ch600[si+(2500*512):si+(5000*512), 1]/128)
plt.figure(0)
plt.plot(d_rfi, label='data')
plt.plot(ups_skf_ch350,label='sk flags')
plt.legend()
plt.grid()
plt.title("ch 350")

plt.figure(1)
plt.plot(d_no_rfi, label='data')
plt.plot(ups_skf_ch600,label='sk flags')
plt.legend()
plt.grid()
plt.title("ch 600")
plt.show()

'''
#sk = np.load('sk_M512_1234_0x.npy')
#plt.figure(0)
#plt.imshow(sk, aspect='auto', origin='lower')
plt.figure(0)
plt.plot(sk[:,5])
plt.plot(sk[:,10])
plt.plot(sk[:,15])
plt.plot(sk[:,20])
plt.plot(sk[:,25])
plt.grid()
plt.show()'''
#sk_flags = np.zeros(np.shape(sk))
'''sk2_flags = np.zeros(np.shape(sk2))

for i in np.arange(np.shape(sk)[0]):
    for j in np.arange(np.shape(sk)[1]):
        if sk[i,j] < 0.86281:
            sk_flags[i,j] = 1

for i in np.arange(np.shape(sk2)[0]):
    for j in np.arange(np.shape(sk2)[1]):
        if sk2[i,j] < 0.776424:
            sk2_flags[i,j] = 1

num_flags_sk = sum(sum(sk_flags))
perc_flags_sk = 100*num_flags_sk/(np.shape(sk)[0]*np.shape(sk)[1])

num_flags_sk2 = sum(sum(sk2_flags))
perc_flags_sk2 = 100*num_flags_sk2/(np.shape(sk2)[0]*np.shape(sk2)[1])

print("1559: ", num_flags_sk, np.shape(sk))
print(" 519: ", num_flags_sk2, np.shape(sk2))'''
#SK = np.load("/home/vereese/git/phd_data/sk_analysis/SK_histograms_5000000.npy")
#SK = np.load("/home/vereese/git/phd_data/sk_analysis/mpi_sk.npy")
"""print(sk)
f_len,t_len = np.shape(sk)
print("SK shape: ", np.shape(sk))

#for i, x in enumerate(sk[0,:]):
#    if isnan(x):
#        continue

for i in np.arange(f_len):
    for j in np.arange(t_len):
        if isnan(sk[i,j]):
            print("zero")
            sk[i,j] = 0

X = np.arange(f_len)
Y = np.arange(t_len)"""

#X, Y = np.meshgrid(X, Y, indexing='ij')
#print("X", np.shape(X))
#print("Y", np.shape(Y))
#print("sk", np.shape(sk))

#surf = ax.plot_surface(X, Y, sk, cmap=cm.jet(sk/np.amax(sk)), linewidth=0, antialiased=False)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
'''plt.figure(0)
plt.imshow(sk,aspect='auto',origin='lower')

plt.figure(1)
plt.plot(sk[800, :])
plt.plot(sk[700, :])
plt.plot(sk[600, :])
plt.plot(sk[500, :])
plt.plot(sk[400, :])
plt.plot(sk[300, :])
# 1557
plt.axhline(0.86281)
plt.axhline(1.17110)
plt.title(perc_flags_sk)
# 519
#plt.axhline(0.776424)
#plt.axhline(1.32275)
plt.grid()

plt.figure(2)
plt.plot(sk2[800, :])
plt.plot(sk2[700, :])
plt.plot(sk2[600, :])
plt.plot(sk2[500, :])
plt.plot(sk2[400, :])
plt.plot(sk2[300, :])
# 519
plt.axhline(0.776424)
plt.axhline(1.32275)
plt.title(perc_flags_sk2)
plt.grid()
plt.show()'''

"""

x = np.arange(0,1,1/10)
plt.figure(1)
counts, bins, patches = plt.hist(SK[300,:],bins=100)
plt.grid()
print(len(bins), len(counts))
plt.figure(2)
plt.semilogy(bins[1:],counts) #bins, histtype='step')

plt.show()
"""