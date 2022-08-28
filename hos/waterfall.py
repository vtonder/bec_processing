import h5py
import numpy as np
import time
import sys

sys.path.append('..')
from constants import start_indices, num_ch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
args = parser.parse_args()

data_file = h5py.File('/net/com08/data6/vereese/' + args.file, 'r')
data = data_file['Data/bf_raw']
data_len = data_file['Data/timestamps'].shape[0]
sub_integrations = int(data_len/(16384*100)) # chunk size
waterfall = np.empty(num_ch, sub_integrations)
t1 = time.time()
for i in np.arange(sub_integrations):
    start = int(i * 16384 * 100)
    end = start +  (16384 * 100)

    if end >= data_len:
        break

    waterfall += np.sum((data[:, start:end, :].astype(np.float)) ** 2, axis=2)

print("took: ", time.time()-t1)

np.save("waterfall_"+args.file[6:10], waterfall)
