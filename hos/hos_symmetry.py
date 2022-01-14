from matplotlib import pyplot as plt
import numpy as np

"""
A script to test symmetry separately 
References:

[1] Appendix A.1 "The bispectrum and its relationship to phase-amplitude coupling"
"""

font2 = {'family': 'STIXGeneral',
         'color': 'g',
         'size': 26,
         'fontweight': 'bold'
         }

font = {'family': 'STIXGeneral',
        'size': 26}
plt.rc('font', **font)


def sample(bi, w_1, w_2):
    in1 = w_1 + 500
    in2 = w_2 + 500
    return bi[in1, in2]


def write_b(bi, w_1, w_2, val):
    row = w_1 + 500
    col = w_2 + 500
    bi[row, col] = val


M = int(1000)
M_2 = int(1000 / 2)
offset = M_2 - 1
full_bispec = np.zeros([M, M], dtype='complex_')
bispectrum = np.zeros([M_2, M_2], dtype='complex_')
bispectrum[100, 100:200] = 100000000
bispectrum[100:150, 200] = 100000000
bispectrum[100:200, 100] = 100000000
bispectrum[200, 100:150] = 100000000
b2 = bispectrum * 0.5
# this taked care of block I
full_bispec[M_2:M, M_2:M] = bispectrum
full_bispec[M_2:0:-1, M_2:0:-1] = bispectrum

for w1 in np.arange(0, 500):
    for w2 in np.arange(0, -500, -1):
        if w1 > -w2:
            write_b(full_bispec, w1, w2, sample(full_bispec, -w1 - w2, w2))
        else:
            write_b(full_bispec, w1, w2, sample(full_bispec, w1, -w1 - w2))

full_bispec[M_2:0:-1, M_2:M] = full_bispec[M_2:M, M_2:0:-1]

# draw symmetry lines
for w1 in np.arange(-500, 500):
    for w2 in np.arange(-500, 500):
        if w1 == w2 or w1 == -w2:
            write_b(full_bispec, w1, w2, 100000000)
        elif w1 == 0 or w2 == 0:
            write_b(full_bispec, w1, w2, 100000000)
        elif w1 == -0.5 * w2 or w1 == -2 * w2:
            write_b(full_bispec, w1, w2, 100000000)
plt.figure(0, figsize=[10, 10])
plt.imshow(np.abs(full_bispec), origin='lower', extent=[-500, 500, -500, 500])
plt.text(300, 150, '1', fontdict=font2)
plt.text(150, 300, '2', fontdict=font2)
plt.text(-100, 400, '3', fontdict=font2)
plt.text(-300, 350, '4', fontdict=font2)
plt.text(-400, 250, '5', fontdict=font2)
plt.text(-400, 50, '6', fontdict=font2)
plt.text(-300, -150, '7', fontdict=font2)
plt.text(-150, -300, '8', fontdict=font2)
plt.text(50, -400, '9', fontdict=font2)
plt.text(250, -400, '10', fontdict=font2)
plt.text(350, -300, '11', fontdict=font2)
plt.text(350, -100, '12', fontdict=font2)
plt.xlabel("$f_1$ [Hz]")
plt.ylabel("$f_2$ [Hz]")
plt.savefig('/home/vereese/Documents/PhD/URSI/sym.png', bbox_inches='tight')
plt.show()
