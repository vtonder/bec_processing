from matplotlib import pyplot as plt
import numpy as np

def sample(bi,w1,w2):
    in1 = w1+500
    in2 = w2+500
    return bi[in1,in2]

def write_b(bi,w1,w2,val):
    row = w1+500
    col = w2+500
    bi[row,col] = val

M=int(1000)
M_2 = int(1000 / 2)
offset = M_2 - 1
full_bispec = np.zeros([M, M], dtype='complex_')
bispectrum = np.zeros([M_2, M_2], dtype='complex_')
bispectrum[100, 100:200] = 100000000
bispectrum[100:150, 200] = 100000000
bispectrum[100:200,100] = 100000000
bispectrum[200,100:150] = 100000000
b2 = bispectrum*0.5
# this taked care of block I
full_bispec[M_2:M,M_2:M] = bispectrum
full_bispec[M_2:0:-1,M_2:0:-1] = bispectrum

for w1 in np.arange(0,500):
    for w2 in np.arange(0,-500,-1):
        if w1 > -w2:
            write_b(full_bispec, w1, w2, sample(full_bispec,-w1-w2,w2))
        else:
            write_b(full_bispec, w1, w2, sample(full_bispec,w1,-w1-w2))

full_bispec[M_2:0:-1,M_2:M] = full_bispec[M_2:M,M_2:0:-1]

plt.figure(0)
plt.imshow(np.abs(full_bispec), origin='lower')
plt.show()