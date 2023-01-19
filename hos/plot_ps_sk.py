import numpy as np
import sys
sys.path.append('../')
from constants import *
from matplotlib import pyplot as plt

spx = np.load('ps_1234_0x.npy')
sp1x = np.load('./1024/ps_M512_1234_0x.npy')
sp2x = np.load('./1024/ps_M1024_1234_0x.npy')
sp3x = np.load('./1024/ps_M2048_1234_0x.npy')

spy = np.load('ps_1234_0y.npy')
sp1y = np.load('./1024/ps_M512_1234_0y.npy')
sp2y = np.load('./1024/ps_M1024_1234_0y.npy')
sp3y = np.load('./1024/ps_M2048_1234_0y.npy')

f2 = 1712 - (freq_resolution / 2)

for i, freq in enumerate(frequencies):
    delay = 10 ** 6 * (dispersion_constant * vela_dm * (1/(f2**2) - 1 / (freq ** 2)))  # us
    num_2_roll = int(np.round(delay / time_resolution))
    spx[i,:] = np.roll(spx[i,:], num_2_roll)
    sp1x[i,:] = np.roll(sp1x[i,:], num_2_roll)
    #sp2x[i,:] = np.roll(sp2x[i,:], num_2_roll)
    #sp3x[i,:] = np.roll(sp3x[i,:], num_2_roll)
    #sp4x[i,:] = np.roll(sp4x[i,:], num_2_roll)
    #sp5x[i,:] = np.roll(sp5x[i,:], num_2_roll)
    spy[i,:] = np.roll(spy[i,:], num_2_roll)
    sp1y[i,:] = np.roll(sp1y[i,:], num_2_roll)
    #sp2y[i,:] = np.roll(sp2y[i,:], num_2_roll)
    #sp3y[i,:] = np.roll(sp3y[i,:], num_2_roll)
    #sp4y[i,:] = np.roll(sp4y[i,:], num_2_roll)
    #sp5y[i,:] = np.roll(sp5y[i,:], num_2_roll)

px = np.sum(spx, axis=0)
p1x = np.sum(sp1x, axis=0)
#p2x = np.sum(sp2x, axis=0)
#p3x = np.sum(sp3x, axis=0)
#p4x = np.sum(sp4x, axis=0)
#p5x = np.sum(sp5x, axis=0)
#
idx_max_px = px.argmax()
idx_max_p1x = p1x.argmax()
#idx_max_p2x = p2x.argmax()
#idx_max_p3x = p3x.argmax()
#idx_max_p4x = p4x.argmax()
#idx_max_p5x = p5x.argmax()

py = np.sum(spy, axis=0)
p1y = np.sum(sp1y, axis=0)
#p2y = np.sum(sp2y, axis=0)
#p3y = np.sum(sp3y, axis=0)
#p4y = np.sum(sp4y, axis=0)
#p5y = np.sum(sp5y, axis=0)

idx_max_py = py.argmax()
idx_max_p1y = p1y.argmax()
idx_max_p2y = p2y.argmax()
idx_max_p3y = p3y.argmax()
idx_max_p4y = p4y.argmax()
idx_max_p5y = p5y.argmax()

mid_vT = int(vela_samples_T/2) 

px = np.roll(px, mid_vT-idx_max_px)
p1x = np.roll(p1x, mid_vT-idx_max_p1x)
#p2x = np.roll(p2x, mid_vT-idx_max_p2x)
#p3x = np.roll(p3x, mid_vT-idx_max_p3x)
##p4x = np.roll(p4x, mid_vT-idx_max_p4x)
##p5x = np.roll(p5x, mid_vT-idx_max_p5x)
#
py = np.roll(py, mid_vT-idx_max_py)
p1y = np.roll(p1y, mid_vT-idx_max_p1y)
#p2y = np.roll(p2y, mid_vT-idx_max_p2y)
#p3y = np.roll(p3y, mid_vT-idx_max_p3y)
#p4y = np.roll(p4y, mid_vT-idx_max_p4y)
#p5y = np.roll(p5y, mid_vT-idx_max_p5y)
sp1x = np.roll(sp1x, mid_vT-idx_max_p1x)
sp1y = np.roll(sp1y, mid_vT-idx_max_p1y)

p = px**2 + py**2
p1 = p1x**2 + p1y**2
#p2 = p2x**2 + p2y**2
#p3 = p3x**2 + p3y**2
#p4 = p4x**2 + p4y**2
#p5 = p5x**2 + p5y**2
#

plt.figure(0)
plt.plot(p/max(p), label='No mitigation')
plt.plot(p1/max(p1), label='M=512')
#plt.plot(p2/max(p2), label='M=1024')
#plt.plot(p3/max(p3), label='M=2048')

#plt.plot(p2/max(p2), label='M=519')
#plt.plot(p3/max(p3), label='M=1557')
#plt.plot(p4/max(p4), label='M=1038')
#plt.plot(p5/max(p5), label='M=2076')
#
#plt.xlim([35000,40000])
plt.xlabel('samples')
plt.title('vela normalised')
plt.legend()
plt.grid()


plt.figure(1)
plt.plot(p1y, label='y')
plt.plot(p1x, label='x')
#plt.plot(p2, label='M=1024')
#plt.plot(p3, label='M=2048')

#plt.plot(p2/max(p2), label='M=519')
#plt.plot(p3/max(p3), label='M=1557')
#plt.plot(p4/max(p4), label='M=1038')
#plt.plot(p5/max(p5), label='M=2076')
#
#plt.xlim([35000,40000])
plt.xlabel('samples')
plt.title('vela non normalised')
plt.legend()
plt.grid()

plt.figure(2)
plt.imshow(sp1x**2+sp1y**2,aspect='auto',origin='lower')
plt.title('spx')

plt.figure(3)
plt.imshow(sp1y, aspect='auto',origin='lower')
plt.title('spy')

'''plt.figure(4)
plt.imshow(sp2y, aspect='auto',origin='lower')
plt.title('M=1024')

plt.figure(5)
plt.imshow(sp3y, aspect='auto',origin='lower')
plt.title('M=2048')'''

plt.show()


