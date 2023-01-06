import h5py
from constants import frequencies, freq_resolution, time_resolution, num_ch, dispersion_constant, vela_T, vela_dm
import time
import numpy as np

#TODO: check if things are not out by a factor of 10e6 changed from freq_resolution being in Hz to MHz
# what functions to process must be configurable
# whether to save or not and what the file should be called

class Pulsar:

    def __init__(self, rotational_frequency, dm, file_name, fp=1):
        self.rotational_frequency = rotational_frequency  # Hz
        self.period = 1 / self.rotational_frequency  # s
        self.dm = dm  # parsec/cm^3
        self.fn = h5py.File(file_name, 'r')
        self.data = self.fn['Data/bf_raw']
        self.fp = fp  # number of periods to fold over ie fold in multiples of 1 or 2 periods, default is 1
        self.samples_T = self.period*10**6 / time_resolution  # number of samples per period
        self.samples_int_T = int(self.samples_T)  # integer number of samples per period
        self.num_data_points = self.data['Data/timestamps'].shape[0]
        self.num_pulses = np.floor(self.num_data_points / self.samples_T)
        self.folded_profile = np.zeros([num_ch, fp * self.samples_int_T])

    def incoherent_dedisperse(self):
        f2 = 1712 - (freq_resolution / 2)
        for i, freq in enumerate(frequencies):

            delay = 10**6*(dispersion_constant * self.dm * (1/f2**2 - 1/freq**2)) # us
            num_2_roll = int(np.round(delay/time_resolution)) # samples
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
        total_integrations = int(self.num_pulses / self.fp)
        for i in np.arange(total_integrations):
            start = int(i * (self.fp * self.samples_T))
            end = int(start + (self.fp * self.samples_int_T))

            if end >= self.num_data_points:
                break

            re = self.data[:, start:end, 0].astype(np.float)
            im = self.data[:, start:end, 1].astype(np.float)
            self.folded_profile += re**2 + im**2


    def rfi_mitigation(self):
        # take the mean and subtract from each channel to rid the RFI
        # TODO: look into using max power, then sigma, then mean statistics to get rid of RFI
        for i in np.arange(num_ch):
            mean = np.mean(self.folded_profile[i, :])
            self.folded_profile[i, :] = self.folded_profile[i, :] - mean

if __name__== '__main__':
    pulsar = Pulsar(vela_T, vela_dm, "/net/com08/data6/vereese/1604641569_wide_tied_array_channelised_voltage_0x.h5")
    pulsar.incoherent_dedisperse()
    t1 = time.time()
    pulsar.frequency_phase_power_spectrum()
    print("Folding took: {0} s".format(time.time()-t1))

    np.save('/home/vereese/phd_data/pulsar/1569/dedispersed_folder_power_spectrum',pulsar.folded_profile)