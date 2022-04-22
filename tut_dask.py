import dask.config
from dask.distributed import Client
import dask.array as da
import numpy as np
import time

dask.config.set(scheduler="threads")
no_freq_channels = 10
period = 10
num_pulses = 100000
tot = da.from_array(np.zeros([no_freq_channels, period]))
#tot = np.zeros([no_freq_channels, period])

phase = np.arange(no_freq_channels)
#print(np.ones([10000,10000]))
m = np.arange(12).reshape(2,6)
print(m)
x = da.from_array(m, chunks=(2,2))
#x = np.arange(8).reshape(2,4)
#print(x)
#y = x[:,2:4] + x[:,0:2]
#print(y)

def cum_sum(x):
    return x[2:4,2:4] + x[2:4,4:6]


#y=x.map_overlap(cum_sum, depth=2, trim=False)
y=x.cumsum(axis=1)


t1=time.time()
print(y.compute())
print(time.time()-t1)


#def fold(x):
#    global tot
    # sum over the number of pulses in observation


def dedisperse():
    global tot
    for i in np.arange(no_freq_channels):
        #print(phase[i])
        tot[i,:] = da.roll(tot[i,:], -phase[i])

# period is 10
def create_dummy_pulsar():
    x = np.zeros(no_freq_channels*period*num_pulses).reshape(no_freq_channels, period*num_pulses)
    # set every 10th element to 1 (10000/10=1000) => there's a 1000 pulses in this dummy data set
    x[:, ::period] = np.ones(num_pulses)
    for i in np.arange(no_freq_channels):
        x[i,:] = np.roll(x[i,:],i)
    return x

t=time.time()
print(t)
da_arrays = []
x = create_dummy_pulsar()
def slicing(x,indx1,indx2):
    return x[:,indx1:indx2]
for i in np.arange(num_pulses):
    #tot += x[:,i*period:(i+1)*period] #.compute()
    da_arrays.append(dask.delayed(slicing)(x,i*period,(i+1)*period)) #.compute()
    #da_arrays.append(da.from_array(x[:,i*period:(i+1)*period].astype(float)))#.compute()

    #tot = da.stack(da_arrays, axis=0)
x = dask.delayed(sum)(da_arrays)
print(time.time()-t)
t=time.time()
print(x.compute(scheduler="threads"))
print(time.time()-t)

def main():
    #client = Client(n_workers=4)
    print("create array")
    t1 = time.time()
    x = da.from_array(create_dummy_pulsar(), chunks=(no_freq_channels,100*period))
    #x = create_dummy_pulsar()
    print("took ", time.time()-t1)
    print("fold")
    t1 = time.time()
    fold(x)
    print("took ", time.time()-t1)
    print("dedisperse")
    t1 = time.time()
    dedisperse()
    print("took ", time.time()-t1)
    print(tot)
    print("calling compute")
    t1 = time.time()
    print(tot.compute())
    print("took ", time.time()-t1)

# remember, __name__ gets set to __main__ when this script is run
# when it's imported __name__ gets set to tut_dask and the if won't run
if not __name__ == "__main__":
   main()