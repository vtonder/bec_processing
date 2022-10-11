from collections import Counter
import h5py
import numpy as np
import argparse
import sys
sys.path.append('..')
from constants import start_indices, time_chunk_size
from matplotlib import pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("file", help="observation file to process. search path: /net/com08/data6/vereese/")
parser.add_argument("ch", help="frequency channel to conduct histogram on")
args = parser.parse_args()

data_file = h5py.File('/net/com08/data6/vereese/' + args.file, 'r') #, rdcc_nbytes=0)
start_index = start_indices[args.file]
data = data_file['Data/bf_raw']
data_len = int((data.shape[1] / time_chunk_size) * time_chunk_size - start_index)
chunks_rank = np.floor(data_len / time_chunk_size / size)  # number of chunks per rank to process, make it a round number
data_len = int(size * chunks_rank * time_chunk_size)  # ensure data_len is a multiple of time_chunk_size
start = int(rank * chunks_rank * time_chunk_size + start_index)
end = int(start + chunks_rank * time_chunk_size)

for idx, i in enumerate(range(start, end, time_chunk_size)):
    d1 = data[int(args.ch), i:(i + time_chunk_size), 0].astype(np.float)
    if idx == 0:
        data_cnt = Counter(d1)
    else:
        data_cnt = data_cnt + Counter(d1)

if rank == 0:
    for i in range(1, size):
        if i == 1:
            data_tmp = comm.recv(source=i, tag=14)
        else:
            data_tmp = data_tmp + comm.recv(source=i, tag=14)

    data_cnt = data_cnt + data_tmp
    pol = args.file[-5:-3]
    fig_name = 'hist_'+args.file[6:10]+'_'+pol+'_'+args.ch

    plt.figure()
    plt.bar(data_cnt.keys(),data_cnt.values())
    plt.savefig(fig_name)
else:
    comm.send(data_cnt, dest=0, tag=14)



