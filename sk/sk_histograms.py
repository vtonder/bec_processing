import numpy as np
from kurtosis import spectral_kurtosis
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("number of processors: ", size)
print("processor: ", rank)

mean = 0
std = 2
FFT_LEN = 1024
M = 512  # number of averages to take PSD over
N = FFT_LEN * M
num_experi = 300000

sk_rank = int(num_experi/size)

SK = np.zeros([FFT_LEN, sk_rank])

for i in np.arange(sk_rank):
    wgn = np.random.normal(mean, std, size=N)
    x = wgn
    SK[:,i] = spectral_kurtosis(x, M, FFT_LEN, normalise=False)

if rank == 0:
    tot_SK = np.zeros([int(FFT_LEN), num_experi])
    tot_SK[:,0:sk_rank] = SK

    for i in range(1, size):
        tmp_sk = np.zeros([FFT_LEN, sk_rank], dtype='float64')
        comm.Recv([tmp_sk, MPI.DOUBLE], source=i, tag=14)
        tot_SK[:, i * sk_rank:(i + 1) * sk_rank] = tmp_sk
    np.save("SK_histograms", SK)
else:
    comm.Send([SK, MPI.DOUBLE], dest=0, tag=14)