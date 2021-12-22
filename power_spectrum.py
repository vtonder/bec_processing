import h5py
import numpy as np
from numba import cuda 
import time
from constants import *
from matplotlib import pyplot as plt

# TODO: find out how to read a chunk and operate on entire chunks at a time
# https://stackoverflow.com/questions/21766145/h5py-correct-way-to-slice-array-datasets

SAVE_DATA = True
COMPUTE_POWER_SPECTA = True 
COMPUTE_TIME_SERIES = False 
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


#vela_x = h5py.File('/home/vereese/pulsar_data/1604641569_wide_tied_array_channelised_voltage_0x.h5', 'r')
#vela_x = h5py.File('/home/vereese/pulsar_data/1604641064_wide_tied_array_channelised_voltage_0y.h5', 'r')
vela_x = h5py.File('/home/vereese/pulsar_data/1604641234_wide_tied_array_channelised_voltage_0x.h5', 'r')

num_data_points = vela_x['Data/timestamps'].shape[0]
print("Number of data points", num_data_points)


t=time.time()
print("read in data t:", t)
data = vela_x['Data/bf_raw'][...] #[()]
data = data[:,11620864:,:] # throw away a lot of 0s to speed up processing. the number was obtained using: np.nonzero(data[123,:,0])
print("done reading in data: ", time.time()-t)

num_data_points = data.shape[1]
num_pulses = int(np.floor(num_data_points / vela_samples_T)) #number of vela pulses per observation
print("number of pulses", num_pulses)
vela_int_samples_T = int(np.floor(vela_samples_T))
fp = 1 #number of vela periods to fold over
summed_profile = np.zeros([no_channels, fp * vela_int_samples_T])
inverted_summed_profile = np.zeros([no_channels, fp * vela_int_samples_T])

#all_data = vela_x['Data']['bf_raw'][()] #.value[:,:,0]
#re = all_data[:,:,0].astype('int16')
#im = all_data[:,:,1].astype('int16')#vela_x['Data']['bf_raw'].value[:,:,1]

#frequencies = np.arange(856+(freq_resolution/1e6)/2,1712+(freq_resolution/1e6)/2,freq_resolution/1e6)
frequencies = np.arange(856,1712,freq_resolution/1e6)
reversed_frequencies = list(reversed(frequencies))
f2 = 1712 -((freq_resolution/1e6)/2)

if DEDISPERSE:
    t = time.time()
    print("dedisperse data t:", t)
    # Note, delaying high frequencies not advancing lower frequencies
    for i, freq in enumerate(frequencies):
        delay = c*vela_dm*(1/(f2**2) - 1/(freq**2)) # ms
        num_2_roll = int(np.round(delay/(time_resolution*1000)))
        #print(freq, delay, num_2_roll)
        data[i,:,0] = np.roll(data[i,:,0], num_2_roll) # re[i,:] = np.roll(re[i,:], num_2_roll)
        data[i,:,1] = np.roll(data[i,:,1], num_2_roll)
    print("done dedispersing data: ", time.time()-t)

#smearing_band = [8.3*10e6 * vela_dm * ((freq_resolution/1e6)/freq**3) for freq in reversed_frequencies] # ms
b_2 = freq_resolution/1e6/2 # half frequency resolution
smearing_band = [c*vela_dm*(1/((freq-b_2)**2) - 1/((freq+b_2)**2)) for freq in reversed_frequencies] # ms
smearing_samples = [int(np.round(smear_time/(time_resolution*1000))) for smear_time in smearing_band]

#smearing_samples.reverse() # because the channels are reversed ie highest frequency is ch0

if CODEDISPERSE:
    k = -2*1j*np.pi*vela_dm/(2.41*10**-4) #negative because we're interested in the inverse transfer function
    t = time.time()
    print("co-dedisperse data t:", t)

    # start with higher frequencies first
    for i, freq in enumerate(reversed_frequencies):
        fir_len = smearing_samples[i]
        number_segments = int(num_data_points/(smearing_samples[i]))
        fft_size = int(2**np.ceil(np.log((smearing_samples[i])+fir_len)/np.log(2)))
        padded_re = np.zeros([number_segments, fft_size])
        padded_im = np.zeros([number_segments, fft_size])
        idx1 = int(fir_len/2)
        idx2 = int(fir_len/2 + smearing_samples[i])
        padded_re[:, idx1:idx2] = data[i,0:number_segments*smearing_samples[i],0].reshape([number_segments, smearing_samples[i]])
        padded_im[:, idx1:idx2] = data[i,0:number_segments*smearing_samples[i],1].reshape([number_segments, smearing_samples[i]])

        padded_complex = padded_re + 1j*padded_im

        # Transfer function of ISM. Note, everything should be in MHz
        ism = np.zeros(fft_size)+1j*np.zeros(fft_size)
        intra_frequencies = np.arange(-(freq_resolution / 1e6) / 2, (freq_resolution / 1e6) / 2, (freq_resolution / 1e6) / fft_size)+freq
        for j, offset in enumerate(intra_frequencies):
            ism[j] = np.exp((k * offset ** 2) / (freq ** 2 * (offset + freq)))

        #print(ism)

        codep_f = np.fft.fft(padded_complex, fft_size) * ism

        codep_t = np.fft.ifft(codep_f, fft_size)

        data[i,0:number_segments*smearing_samples[i],0] = codep_t[:,idx1:idx2].reshape(number_segments*smearing_samples[i]).real
        data[i,0:number_segments*smearing_samples[i],1] = codep_t[:,idx1:idx2].reshape(number_segments*smearing_samples[i]).imag
    print("done co-dedispersing data: ", time.time()-t)

if MEAS_RESIDUAL_DISP:
    t1 = time.time()
    print("measuring residual dispersion", t1)

    fine_fft_len = int(64)
    highest_chan = int(1010) 
    num_chans = int(3)
    meas_chs = np.arange(highest_chan,highest_chan-num_chans,-1)

    new_freq_res = freq_resolution/fine_fft_len
    new_time_res = time_resolution*fine_fft_len
    print("new time res", new_time_res)
    new_vela_samples_T = vela_T / new_time_res
    new_vela_int_samples_T = int(new_vela_samples_T)

    dp = len(data[meas_chs[0],:,0])
    seg = int(dp/fine_fft_len)
    new_num_pulses = int(np.floor(seg / new_vela_samples_T))
    print("segments", seg)
    print("new number of pulses (should be same)" , new_num_pulses)

    redisp_summed_profile = np.zeros([int(num_chans*fine_fft_len/2), new_vela_int_samples_T])

    for i, meas_ch in enumerate(meas_chs):
        print("populate frequency channels: ", int(i*fine_fft_len/2), int((i+1)*fine_fft_len/2))
        print("processing frequency: ", meas_ch)
        re = data[meas_ch,:,0].reshape(seg, fine_fft_len)
        im = data[meas_ch,:,1].reshape(seg, fine_fft_len)

        fine_pfb = np.fft.fft(re + 1j*im, fine_fft_len)
        fine_pfb = fine_pfb.transpose()

        print("Fine PFB shape: ", fine_pfb.shape)

        #fine_pfb_re = fine_pfb.reshape(fine_fft_len*seg).real
        #fine_pfb_im = fine_pfb.reshape(fine_fft_len*seg).real

        #fine_pfb_im = np.fft.fft(im, fine_fft_len)

        for j in np.arange(new_num_pulses):

            start = int(j*new_vela_samples_T)
            end = start + new_vela_int_samples_T

            if end >= seg:
                break
            # only take 1 half of the spectrum
            #re1 = data[meas_ch,start:end,0].astype(np.float)
            #im1 = data[meas_ch,start:end,1].astype(np.float)
            magnitude = np.abs(fine_pfb[0:int(fine_fft_len/2), start:end]*np.conj(fine_pfb[0:int(fine_fft_len/2), start:end]))
            redisp_summed_profile[int(i*fine_fft_len/2):int((i+1)*fine_fft_len/2), :] += magnitude.real #re**2 + im**2
            #redisp_summed_profile[:,i] += re1**2 + im1**2

        if i == 0:
            num_2_roll = 0
        else:
            freq = reversed_frequencies[highest_chan]
            print("reference frequency for roll: ", freq)
            f2 = reversed_frequencies[highest_chan-i]
            print("f2: ", f2)

            delay = c*vela_dm*(1/(freq**2) - 1/(f2**2)) # ms
            num_2_roll = int(np.round(delay/(new_time_res*2*1000)))
        print(num_2_roll)        
        #print(redisp_summed_profile[int(i*fine_fft_len/2):int(i*fine_fft_len/2)+5, 0:5])
        redisp_summed_profile[int(i*fine_fft_len/2):int((i+1)*fine_fft_len/2), :] = np.roll(redisp_summed_profile[int(i*fine_fft_len/2):int((i+1)*fine_fft_len/2), :], -num_2_roll)
        #print(redisp_summed_profile[int(i*fine_fft_len/2):int(i*fine_fft_len/2)+5, 0:5])


    print("done measuring residula dispersion: ", time.time()-t1)
    if SAVE_DATA:
        np.save('co_residual_summed_profile_test2', redisp_summed_profile)



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



if COMPUTE_POWER_SPECTA:
    tot_int = int(num_pulses/fp)
    rem = 0
    t1=time.time()
    print("fold data", t1)

    for i in np.arange(tot_int):

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

        #t2=time.time()
        #diff = t2-t1
        #print('at addition: ', i, 'of', tot_int ,'took ',  diff, 's')
    print("done fold data", time.time()-t1)


    # take the mean and subtract from each channel to rid the RFI
    # TODO: look into using max power, then sigma, then mean statistics to get rid of RFI
    # invert the channels because we are working with filterbank data
    # In filterbank data the ch0 corresponds to higher frequency components and the higher channels correspond to lower frequencies
    for i in np.arange(no_channels):
        mean = np.mean(summed_profile[i,:])
        #summed_profile[i,:] = summed_profile[i,:]-mean
        inverted_summed_profile[(no_channels-1)-i,:] = summed_profile[i,:]-mean


    if SAVE_DATA:
        np.save('summed_profile_1234', inverted_summed_profile)
