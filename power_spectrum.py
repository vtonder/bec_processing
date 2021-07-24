import h5py
import numpy as np
from numba import cuda 
import time
from constants import *
from matplotlib import pyplot as plt

# TODO: find out how to read a chunk and operate on entire chunks at a time
# https://stackoverflow.com/questions/21766145/h5py-correct-way-to-slice-array-datasets

PLOT = True 
SAVE_DATA = True 
PROCESS_DATA = True 
COMPUTE_POWER_SPECTA = True 
COMPUTE_TIME_SERIES = True 
DEDISPERSE = True 
CODEDISPERSE = True

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
    vela_x = h5py.File('/home/vereese/pulsar_data/1604641569_wide_tied_array_channelised_voltage_0x.h5', 'r') #driver core makes it really slow here
    #vela_x = h5py.File('/home/vereese/pulsar_data/1604641064_wide_tied_array_channelised_voltage_0y.h5', 'r')
    #vela_x = h5py.File('/home/vereese/pulsar_data/1604641234_wide_tied_array_channelised_voltage_0x.h5', 'r', driver="core")

    num_data_points = vela_x['Data/timestamps'].shape[0]
    print("Number of data points", num_data_points)
    num_pulses = int(np.floor(num_data_points / vela_samples_T)) #number of vela pulses per observation
    vela_int_samples_T = int(np.floor(vela_samples_T))
    fp = 1 #number of vela periods to fold over
    
    summed_profile = np.zeros([no_channels, fp * vela_int_samples_T])
    inverted_summed_profile = np.zeros([no_channels, fp * vela_int_samples_T])

    t=time.time()
    print("read in data t:", t)
    data = vela_x['Data/bf_raw'][...]
    t=time.time()
    print("done reading in data: ", t)

    #all_data = vela_x['Data']['bf_raw'][()] #.value[:,:,0]
    #re = all_data[:,:,0].astype('int16')
    #im = all_data[:,:,1].astype('int16')#vela_x['Data']['bf_raw'].value[:,:,1]
    
    frequencies = np.arange(856+(freq_resolution/1e6)/2,1712+(freq_resolution/1e6)/2,freq_resolution/1e6)
    f2 = 1712+((freq_resolution/1e6)/2)
    if DEDISPERSE:
        for i, freq in enumerate(frequencies):
            delay = c*vela_dm*(1/(f2**2) - 1/(freq**2))
            num_2_roll = int(np.round(delay/(time_resolution*1000)))
            print(freq, delay, num_2_roll)
            data[i,:,0] = np.roll(data[i,:,0], num_2_roll) # re[i,:] = np.roll(re[i,:], num_2_roll)
            data[i,:,1] = np.roll(data[i,:,1], num_2_roll)

    if CODEDISPERSE:
        k = -2*np.pi*freq_resolution**2*vela_dm/(2.41*10**-4)
        for i, freq in enumerate(frequencies):
            ism = np.exp(k/(freq**2*(freq_resolution+freq)))
            data[i,:,:] = data[i,:,:] * ism

    if COMPUTE_TIME_SERIES:
        # randomly chose to integrate 22 vela pulses per sub-integration
        num_sub_ints = int(200) # number of sub-integrations
        step = int(vela_int_samples_T * num_sub_ints)
        num_int = int(np.floor(num_data_points/step)) # total number of integrations
        vela_sub_int = np.zeros([num_int, vela_int_samples_T])
        fc = 512 
        for i in np.arange(num_int):
            print("at integration ", i, " of ", num_int)
            for j in np.arange(num_sub_ints):
                start = (i*step)+(j * vela_int_samples_T)
                stop = (i*step)+((j+1) * vela_int_samples_T)
                re = data[fc, start:stop, 0].astype(np.float)
                im = data[fc, start:stop, 1].astype(np.float)
                vela_sub_int[i,:] += re**2 + im**2

        if SAVE_DATA:
            np.save('sub_integration_true_period_1569', vela_sub_int)      
            
        if PLOT:
            plt.figure()
            plt.autoscale(True)
            plt.ylabel('sub integrations')
            plt.xlabel('Pulsar phase')
            plt.imshow(vela_sub_int, aspect='auto')
            plt.show()

    if COMPUTE_POWER_SPECTA:
        tot_int = int(num_pulses/fp)
        rem = 0
        for i in np.arange(tot_int):
            t1=time.time()
            print(t1)
        
            #add(vela_x['Data']['bf_raw'][ch,i*74670:(i+1)*74670,0], vela_x['Data']['bf_raw'][ch,i*74670:(i+1)*74670,1], temp)
            #summed_profile += re[:,i*(fp*vela_samples_T):(i+1)*(fp*vela_samples_T)]**2 + im[:,i*(fp*vela_samples_T):(i+1)*(fp*vela_samples_T)]**2
            #vela_x['Data/bf_raw'].read_direct(data, source_sel=np.s_[:,i*(fp*vela_samples_T):(i+1)*(fp*vela_samples_T),:])
            #vela_x['Data']['bf_raw'].read_direct(im, source_sel=np.s_[:,i*(fp*vela_samples_T):(i+1)*(fp*vela_samples_T),1])

            start = int(i*(fp * vela_samples_T))
            end = start + (fp*vela_int_samples_T) # int(np.floor((i + 1) * (fp * vela_samples_T)))

            if end >= num_data_points:
                break

            re = data[:,start:end, 0].astype(np.float)
            im = data[:,start:end, 1].astype(np.float)

            summed_profile += re**2 + im**2

            t2=time.time()
            diff = t2-t1
            print('at addition: ', i, 'of', tot_int ,'took ',  diff, 's')
        
        # take the mean and subtract from each channel to rid the RFI
        # TODO: look into using max power, then sigma, then mean statistics to get rid of RFI
        # invert the channels because we are working with filterbank data
        # In filterbank data the ch0 corresponds to higher frequency components and the higher channels correspond to lower frequencies
        for i in np.arange(no_channels):
            #mean = np.mean(summed_profile[i,:])
            #summed_profile[i,:] = summed_profile[i,:]-mean
            inverted_summed_profile[(no_channels-1)-i,:] = summed_profile[i,:]#-mean


        if SAVE_DATA:
            np.save('summed_profile_1569', inverted_summed_profile)      

if PLOT: 
    summed_profile = np.load('summed_profile_1569.npy')
    for i in np.arange(no_channels):
        mean = np.mean(summed_profile[i,:])
        summed_profile[i,:] = summed_profile[i,:]-mean

    plt.figure(0)
    #plt.autoscale(True)
    plt.imshow(summed_profile, aspect='auto', extent=[0,1,856,1712])#, origin='lower')
    #plt.imshow(summed_profile, aspect='auto')
    #plt.plot(summed_profile[292,:])
    plt.ylabel('Frequency [MHz]')
    plt.xlabel('Pulsar phase')
    #plt.imshow(np.roll(summed_profile, -30000, axis=1), aspect='auto', extent=[0,1,856,1712]) #interpolation='nearest'
    #plt.colorbar()
    #plt.clim([-0.5,7000])
    #plt.plot(summed_profile[100,:])
    plt.show()

    vela_sub_int = np.load('sub_integration_true_period_1569.npy')
    print(np.shape(vela_sub_int[:,0:2500]))
    print(np.shape(vela_sub_int))



    plt.figure(1)
    plt.autoscale(True)
    plt.ylabel('sub integrations')
    plt.xlabel('Pulsar phase')
    plt.imshow(vela_sub_int[:,10000:25000], aspect='auto', origin='lower')
    plt.show()
