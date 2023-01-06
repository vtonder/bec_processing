import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from math import isnan

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#sk = np.load("/home/vereese/git/phd_data/sk_analysis/1234_0x_sk.npy")
SK = np.load("/home/vereese/git/phd_data/sk_analysis/SK_histograms_5000000.npy")
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
"""plt.figure(0)
plt.imshow(sk,aspect='auto',origin='lower')

plt.figure(1)
plt.plot(sk[:, 420])
plt.grid()
plt.show()"""

x = np.arange(0,1,1/10)
plt.figure(1)
counts, bins, patches = plt.hist(SK[300,:],bins=100)
plt.grid()
print(len(bins), len(counts))
plt.figure(2)
plt.semilogy(bins[1:],counts) #bins, histtype='step')

plt.show()
