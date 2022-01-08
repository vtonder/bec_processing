from matplotlib import pyplot as plt
import numpy as np

"""
A script to test symmetry separately 
References:

[1] Appendix A.1 "The bispectrum and its relationship to phase-amplitude coupling"
"""

font2 = {'family': 'STIXGeneral',
        'color':  'g',
        'size': 24,
        'fontweight':'bold'
        }

font = {'family': 'STIXGeneral',
        'size': 24}
plt.rc('font', **font)

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

#draw symmetry lines
for w1 in np.arange(-500,500):
    for w2 in np.arange(-500,500):
        if w1 == w2 or w1 == -w2:
            write_b(full_bispec, w1, w2,100000000)
        elif w1 == 0 or w2 == 0:
            write_b(full_bispec, w1, w2,100000000)
        elif w1 == -0.5*w2 or w1 == -2*w2:
            write_b(full_bispec, w1, w2, 100000000)
fig,ax =plt.subplots()
ax.imshow(np.abs(full_bispec), origin='lower')
plt.text(800, 650, '1', fontdict=font2)
plt.text(650, 800, '2', fontdict=font2)
plt.text(400, 900, '3', fontdict=font2)
plt.text(200, 850, '4', fontdict=font2)
plt.text(100, 750, '5', fontdict=font2)
plt.text(100, 550, '6', fontdict=font2)
plt.text(200, 350, '7', fontdict=font2)
plt.text(350, 200, '8', fontdict=font2)
plt.text(550, 100, '9', fontdict=font2)
plt.text(750, 100, '10', fontdict=font2)
plt.text(850, 200, '11', fontdict=font2)
plt.text(850, 400, '12', fontdict=font2)
plt.tick_params(axis='both', which='both', length=0)
ax.set(xlabel='$\omega_1$',ylabel='$\omega_2$')
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
ax.tick_params(axis='both', which='both', length=0)
fig.subplots_adjust(left=0,right=1,bottom=0.01,top=1)
plt.show()