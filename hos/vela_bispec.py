from mpi4py import MPI
import h5py
import numpy as np
from constants import num_ch, start_indices, xy_time_offsets, pulsars, time_chunk_size
from hos import Bispectrum
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    t1 = time.time()

f = '1604641234_wide_tied_array_channelised_voltage_0x.h5'
df = h5py.File('/net/com08/data6/vereese/' + f, 'r')
si = start_indices[f] + xy_time_offsets[f] # start index of x polarisation
ndp = df['Data/timestamps'].shape[0] - si # number of data points, x pol

nd_rank = np.floor(ndp / rank / time_chunk_size) * rank * time_chunk_size
nc_rank = nd_rank/time_chunk_size
start = rank*nd_rank
bico_I = np.zeros([512,512], dtype=np.csingle)
for i in np.arange(start, start+nd_rank, time_chunk_size):
    data = df['Data/bf_raw'][:,i:i+time_chunk_size,:].astype(np.float16)
    data_c = np.transpose(data[:,:,0] + 1j*data[:,:,1])
    b = Bispectrum(data_c, reshape=False, fft_size=1024, method='direct')
    bi = b.direct_bispectrum(compute_fft=False)
    bico_I = bico_I + bi

bico_I = np.abs(bico_I/nc_rank)

if rank > 0:
    comm.Send([bico_I, MPI.DOUBLE], dest=0, tag=15)  # send results to process 0
else:
    for i in range(1, size):
        tmp_bico = np.zeros([512,512], dtype=np.csingle)
        comm.Recv([tmp_bico, MPI.DOUBLE], source=i, tag=15)
        bico_I += np.float32(tmp_bico)

    bico_I = np.abs(bico_I/size)

    np.save('vela_bispectrum', bico_I)
    print("processing took: ", time.time() - t1)
