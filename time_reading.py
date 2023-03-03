import time
import h5py

input_file = h5py.File('/home/vereese/pulsar_data/1604641569_wide_tied_array_channelised_voltage_0x.h5', 'r')
#input_file = h5py.File('/home/vereese/pulsar_data/1604641064_wide_tied_array_channelised_voltage_0y.h5', 'r')

print("Using elipses [...]")
t1 = time.time()
data = input_file['Data/bf_raw'][...]
print("took :", time.time()-t1)


print("Using [()]")
t1 = time.time()
data = input_file['Data/bf_raw'][()]
print("took :", time.time()-t1)

