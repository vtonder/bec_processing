import h5py
import numpy as np
from matplotlib import pyplot as plt

df = h5py.File('/net/com08/data6/vereese/1604641234_wide_tied_array_channelised_voltage_0x.h5', 'r')
dfy = h5py.File('/net/com08/data6/vereese/1604641234_wide_tied_array_channelised_voltage_0y.h5', 'r')


'''plt.figure(0)
plt.plot(df['Data/bf_raw'][150,13631488:14741488,0], label='ch 150')
plt.plot(df['Data/bf_raw'][600,13631488:14741488,0], label='ch 600')
plt.grid()
plt.legend()

plt.figure(1)
plt.plot(dfy['Data/bf_raw'][150,46767104:47867104,0], label='ch 150')
plt.plot(dfy['Data/bf_raw'][600,46767104:47867104,0], label='ch 600')
plt.grid()
plt.legend()'''

plt.figure(0)
plt.plot(df['Data/bf_raw'][150,13631488+593750:13631488+595100,0], label='ch 150')
plt.plot(df['Data/bf_raw'][600,13631488+626500:13631488+627850,0], label='ch 600')
plt.ylim([120, 127.5])
plt.grid()
plt.legend()


plt.show()
