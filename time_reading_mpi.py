import h5py
from mpi4py import MPI
from constants import time_chunk_size
import time
import argparse
import numpy as np

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument("tag", help="observation tag to process. search path: /net/com08/data6/vereese/")
args = parser.parse_args()

f = '160464' + args.tag + '_wide_tied_array_channelised_voltage_0x.h5'
if rank == 0:
    t_avr = 0
for i in np.arange(20):
    t1 = time.time()
    with h5py.File('/net/com08/data6/vereese/' + f, 'r') as df:
        
        tot_ndp = df['Data/timestamps'].shape[0] # total number of data points of x polarisation
        data_rank = int(tot_ndp/size) 
        chunk_rank = int(data_rank/time_chunk_size)
        
        offset = rank*chunk_rank*time_chunk_size
        for i in np.arange(chunk_rank):
            start = offset + i*time_chunk_size
            stop = start + time_chunk_size
            data = df['Data/bf_raw'][:,start:stop,:]
    td = time.time()-t1
    t = comm.gather(td, root=0)
    if rank == 0:
        t = np.asarray([t])
        t_avr = t_avr + t.sum()/size

if rank == 0:
    print("size: ", size)
    print("processing time: ", t_avr/20)


