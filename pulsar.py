import h5py
from constants import frequencies, freq_resolution, time_resolution, num_ch, dispersion_constant
import time
import numpy as np

#TODO: check if things are not out by a factor of 10e6 changed from freq_resolution being in Hz to MHz
# what functions to process must be configurable
# whether to save or not and what the file should be called

class Pulsar:

    def __init__(self, period, dm, file_name, fp):
        self.period = period  # is
        self.frequency = 1 / period  # Hz
        self.dm = dm  # to be applied to ms and parsec/cm^3
        self.data = h5py.File(file_name, 'r')
        self.fp = fp  # number of periods to fold over
        self.samples_T = period / time_resolution  # number of samples per period
        self.samples_int_T = int(self.samples_T)  # integer number of samples per period
        self.num_data_points = self.data['Data/timestamps'].shape[0]
        self.num_pulses = np.floor(self.num_data_points / self.samples_T)
        self.folded_profile = np.zeros([num_ch, fp * self.samples_int_T])
        self.inverted_folded = np.zeros([num_ch, fp * self.samples_int_T])

    def incoherent_dedisperse(self):
        f2 = 1712 + (freq_resolution / 2)
        for i, freq in enumerate(frequencies):
            # delay = c*vela_dm*(1/(f2**2) - 1/(freq**2))
            delay = dispersion_constant * self.dm * (1 / (f2 ** 2) - 1 / (freq ** 2))

            num_2_roll = int(np.round(delay / (time_resolution * 1000)))
            print(freq, delay, num_2_roll)
            # re[i,:] = np.roll(re[i,:], num_2_roll)
            # im[i,:] = np.roll(im[i,:], num_2_roll)
            self.data[i, :, 0] = np.roll(self.data[i, :, 0], num_2_roll)
            self.data[i, :, 1] = np.roll(self.data[i, :, 1], num_2_roll)

    def sub_integration_plot(self):
        # randomly chose to integrate 22 vela pulses per sub-integration
        num_sub_ints = int(100)  # number of sub-integrations
        # step = int(vela_int_samples_T * num_sub_ints)
        step = int(self.samples_int_T * num_sub_ints)
        num_int = int(np.floor(self.num_data_points / step))  # total number of integrations
        # vela_sub_int = np.zeros([num_int, vela_int_samples_T])
        J0742_sub_int = np.zeros([num_int, self.samples_int_T])

        fc = 512
        for i in np.arange(num_int):
            print("at integration ", i, " of ", num_int)
            for j in np.arange(num_sub_ints):
                # start = (i*step)+(j * vela_int_samples_T)
                # stop = (i*step)+((j+1) * vela_int_samples_T)
                start = (i * step) + (j * self.samples_int_T)
                stop = (i * step) + ((j + 1) * self.samples_int_T)
                re = self.data[fc, start:stop, 0].astype(np.float)
                im = self.data[fc, start:stop, 1].astype(np.float)
                # vela_sub_int[i,:] += re**2 + im**2
                J0742_sub_int[i, :] += re ** 2 + im ** 2

            np.save('sub_int_', J0742_sub_int)

    def frequency_phase_power_spectrum(self):
        tot_int = int(self.num_pulses / self.fp)
        for i in np.arange(tot_int):
            t1 = time.time()
            print(t1)

            # add(vela_x['Data']['bf_raw'][ch,i*74670:(i+1)*74670,0], vela_x['Data']['bf_raw'][ch,i*74670:(i+1)*74670,1], temp)
            # summed_profile += re[:,i*(fp*vela_samples_T):(i+1)*(fp*vela_samples_T)]**2 + im[:,i*(fp*vela_samples_T):(i+1)*(fp*vela_samples_T)]**2
            # vela_x['Data/bf_raw'].read_direct(data, source_sel=np.s_[:,i*(fp*vela_samples_T):(i+1)*(fp*vela_samples_T),:])
            # vela_x['Data']['bf_raw'].read_direct(im, source_sel=np.s_[:,i*(fp*vela_samples_T):(i+1)*(fp*vela_samples_T),1])

            # start = int(i*(fp * vela_samples_T))
            # end = start + vela_int_samples_T

            start = int(i * (self.fp * self.samples_T))
            end = int(start + (self.fp * self.samples_int_T))

            # end = int((i + 1) * (fp * vela_samples_T))

            if end >= self.num_data_points:
                break

            re = self.data[:, start:end, 0].astype(np.float)
            im = self.data[:, start:end, 1].astype(np.float)

            # summed_profile += re[:,start:end]**2 + im[:,start:end]**2
            self.folded_profile += re ** 2 + im ** 2
            t2 = time.time()
            diff = t2 - t1
            print('at addition: ', i, 'of', tot_int, 'took ', diff, 's')

        # take the mean and subtract from each channel to rid the RFI
        # TODO: look into using max power, then sigma, then mean statistics to get rid of RFI
        # invert the channels because we are working with filterbank data
        # In filterbank data the ch0 corresponds to higher frequency components and the higher channels correspond to lower frequencies
        for i in np.arange(num_ch):
            mean = np.mean(self.folded_profile[i, :])
            self.folded_profile[i, :] = self.folded_profile[i, :] - mean
            self.inverted_folded[(num_ch - 1) - i, :] = self.folded_profile[i, :] - mean
