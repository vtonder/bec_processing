from collections import Counter
import h5py
import numpy as np
import argparse
import sys
sys.path.append('..')
from constants import start_indices
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("ch", help="frequency channel to conduct histogram on")
args = parser.parse_args()

data_file = h5py.File('/net/com08/data6/vereese/' + args.file, 'r') #, rdcc_nbytes=0)
start_index = start_indices[args.file]
data = data_file['Data/bf_raw'][int(args.ch), start_index:, 0]

data_cnt = Counter(data)
pol = args.file[-5:-3]
fig_name = 'hist_'+args.file[6:10]+'_'+pol+'_'+args.ch

plt.figure()
plt.bar(data_cnt.keys(),data_cnt.values())
plt.savefig(fig_name)


