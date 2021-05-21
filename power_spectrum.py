import h5py
import numpy as np
from numba import cuda 
import time
from constants import *
from matplotlib import pyplot as plt

# TODO: find out how to read a chunk and operate on entire chunks at a time

PLOT = True 
SAVE_DATA = True 
PROCESS_DATA = True #False 
DEDISPERSE = True #False 

#@cuda.jit#nopython=True)
#def square(data, y):
#    y = data*data

@cuda.jit#(device=True)#nopython=True)
def add(re, im, x):
    x = re*re + im*im 
    #return x 


#@numba.jit(nopython=True)
#def accumulate(summed_data, data, ch):
#    summed_data[ch][:] += data
#    return summed_data
font = {'family' : 'STIXGeneral',
        'size'   : 22}

        #'weight' : 'bold',
        
plt.rc('font', **font)


if PROCESS_DATA:
    #vela_x = h5py.File('/home/vereese/pulsar_data/1604641569_wide_tied_array_channelised_voltage_0x.h5', 'r')
    #vela_x = h5py.File('/home/vereese/pulsar_data/1604641064_wide_tied_array_channelised_voltage_0y.h5', 'r')
    vela_x = h5py.File('/home/vereese/pulsar_data/1604641234_wide_tied_array_channelised_voltage_0x.h5', 'r')

    num_data_points = vela_x['Data']['timestamps'].shape[0]
    print(num_data_points)
    num_pulses = int(np.floor(num_data_points/vela_samples_T)) #number of vela pulses per observation
    vela_samples_T = int(np.floor(vela_samples_T))
    fp = 1 #number of vela periods to fold over
    
    print("read in data")
    
    summed_profile = np.zeros([no_channels, fp*vela_samples_T])
    inverted_summed_profile = np.zeros([no_channels, fp*vela_samples_T])

    #temp = np.zeros([no_channels, fp*vela_samples_T])
    
    t=time.time()
    print("reading all: ", t)
    all_data = vela_x['Data']['bf_raw'][()] #.value[:,:,0]
    
    t=time.time()
    print("getting re ", t)
    re = all_data[:,:,0].astype('int16')
    
    t=time.time()
    print("getting im ", t)
    im = all_data[:,:,1].astype('int16')#vela_x['Data']['bf_raw'].value[:,:,1]
    
    #for i in np.arange(100,(int(num_pulses/fp))-100):
    frequencies = np.arange(1712+(freq_resolution/1e6)/2,856+(freq_resolution/1e6)/2,-freq_resolution/1e6)
    f2 = 1712+((freq_resolution/1e6)/2)
    if DEDISPERSE:
        for i, freq in enumerate(frequencies):
            delay = c*vela_dm*(1/(f2**2) - 1/(freq**2))
            num_2_roll = int(np.round(delay/(time_resolution*1000)))
            print(freq, delay, num_2_roll)
            re[i,:] = np.roll(re[i,:], num_2_roll)
            im[i,:] = np.roll(im[i,:], num_2_roll)

    for i in np.arange(int(num_pulses/fp)):

        t1=time.time()
        print(t1)
    
        #add(vela_x['Data']['bf_raw'][ch,i*74670:(i+1)*74670,0], vela_x['Data']['bf_raw'][ch,i*74670:(i+1)*74670,1], temp)
        summed_profile += re[:,i*(fp*vela_samples_T):(i+1)*(fp*vela_samples_T)]**2 + im[:,i*(fp*vela_samples_T):(i+1)*(fp*vela_samples_T)]**2
    
        t2=time.time()
        diff = t2-t1
        print('at addition: ', i, 'took ',  diff, 's')
    
    # take the mean and subtract from each channel to rid the RFI
    # TODO: look into using max power, then sigma, then mean statistics to get rid of RFI
    # invert the channels because we are working with filterbank data
    # In filterbank data the ch0 corresponds to higher frequency components and the higher channels correspond to lower frequencies
    for i in np.arange(no_channels):
        mean = np.mean(summed_profile[i,:])
        #summed_profile[i,:] = summed_profile[i,:]-mean
        inverted_summed_profile[(no_channels-1)-i,:] = summed_profile[i,:]-mean

    #frequencies = np.arange(856+(freq_resolution/1e6)/2,1712+(freq_resolution/1e6)/2,-freq_resolution/1e6)
    #f2 = 1284.0+((freq_resolution/1e6)/2)
    #if DEDISPERSE:
    #    for i, freq in enumerate(frequencies):
    #        delay = c*vela_dm*(1/(freq**2) - 1/(f2**2))
    #        num_2_roll = int(np.round(delay/(time_resolution*1000)))
    #        print(freq, delay, num_2_roll)
    #        inverted_summed_profile[i,:] = np.roll(inverted_summed_profile[i,:], num_2_roll)
    #        #re[i,:] = np.roll(re[i,:], num_2_roll)
    #        #im[i,:] = np.roll(im[i,:], num_2_roll)


    if SAVE_DATA:
        np.save('summed_profile', inverted_summed_profile)      

if PLOT: 
    summed_profile = np.load('summed_profile.npy')
    plt.figure()
    plt.autoscale(True)
    #plt.imshow(summed_profile, aspect='auto', extent=[0,1,856,1712])#, origin='lower')
    #plt.imshow(summed_profile, aspect='auto')
    plt.ylabel('Frequency [MHz]')
    plt.xlabel('Pulsar phase')
    plt.imshow(np.roll(summed_profile, -30000, axis=1), aspect='auto', extent=[0,4,856,1712]) #interpolation='nearest'
    #plt.colorbar()
    #plt.clim([-0.5,7000])
    #plt.plot(summed_profile[100,:])
    plt.show()

