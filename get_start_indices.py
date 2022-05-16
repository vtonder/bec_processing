import sys
import h5py
import numpy as np

# The meerkat data has lots of 0s
# This script takes an h5py input file 
# It calculates the first index to be non 0 per channel
# The ultimate index chosen is the maximum index between all the channels for re and imaginary

DIRECTORY = '/home/vereese/pulsar_data/'

file_name = sys.argv[1]
input_data = h5py.File(DIRECTORY+file_name, 'r')
data = input_data['Data/bf_raw'][...]
data_re = data[:,:,0]
data_im = data[:,:,1]
print("data len re: ", len(data_re)) # Re & Im freq channels will have the same length
print("data len im: ", len(data_im))
re_start, im_start = [], []

# Throw away 55 ch on each side given receiver band is 900-1670 MHz
for k in np.arange(55,969):
    print(k)
    re_start.append(next(i for i,j in enumerate(data_re[k]) if j != 0))
    im_start.append(next(i for i,j in enumerate(data_im[k]) if j != 0)) 


print("re ", max(re_start))
print("im ", max(im_start))

