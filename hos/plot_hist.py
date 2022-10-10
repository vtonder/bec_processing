from collections import Counter
import h5py
import numpy as np
import argparse
from constants import start_indices
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("ch", help="frequency channel to conduct histogram on")
args = parser.parse_args()

data_file = h5py.File('/net/com08/data6/vereese/' + args.file, 'r', rdcc_nbytes=0)
start_index = start_indices[args.file]
data = Counter(data_file['Data/bf_raw'][args.ch, start_index:, 0])

plt.figure()
plt.bar(data.keys(),data.values())
plt.show()


