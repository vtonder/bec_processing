import h5py
import numpy as np
import time
import sys
sys.path.append('../')
from constants import num_ch, start_indices, lower_limit, upper_limit, pulsars 
sys.path.append('../pulsar_processing')
import argparse

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("-m", dest="M", help="Number of spectra to accumulate in SK calculation", default=512)
args = parser.parse_args()

df = h5py.File('/net/com08/data6/vereese/' + args.file, 'r')
start_index = start_indices[args.file]
num_data_points = df['Data/timestamps'].shape[0] - start_index
data = df['Data/bf_raw'][:,int(start_index):,:]
tag = args.file[6:10] + '_'   # add last 4 digits of observation code onto the file_name
pol = args.file[-5:-3]  # polarisation 0x or 0y

pulsar = pulsars[tag[:-1]] 
samples_T = pulsar['samples_T']
int_samples_T = int(np.floor(samples_T))

num_pulses = int(np.floor(num_data_points / samples_T))  # number of vela pulses per observation

summed_profile = np.zeros([num_ch, int_samples_T])
for i in np.arange(num_pulses):
    start = int(i * samples_T)
    end = start + int_samples_T
    if end >= num_data_points:
        break
    re = data[:, start:end, 0].astype(np.float)/128
    im = data[:, start:end, 1].astype(np.float)/128
    summed_profile += re ** 2 + im ** 2

np.save('ps_' + tag + pol, summed_profile)
print("procesing took: ", time.time() - t1)

