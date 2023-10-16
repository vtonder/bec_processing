import numpy as np
from constants import pulsars, num_ch, frequencies, dispersion_constant, freq_resolution, time_resolution
class PulsarIntensity:
    def __init__(self, dir, tag, file_name, sf = None):
        """

        :param dir: location of data
        :param tag:
        :param file_name: folded pulsar intensity file. freq channels X pulse num samples
        :param sf: summed flags
        """
        self.file_name = file_name
        self.tag = tag
        self.I = np.load(dir + file_name)
        self.pulsar = pulsars[self.tag]
        self.samples_T = int(np.floor(self.pulsar['samples_T']))
        self.num_pol = 2 # H and V polarisation
        if sf:
            self.sf = np.load(dir + sf)
            self.rfi = 100*(np.float32(self.sf).sum(axis=1) / (self.num_pol*self.samples_T*32*1413))
        else:
            self.sf = None
        self.pulse_start = 0
        self.pulse_stop = 0
        self.pulse_width = 0

    def apply_mask(self):
        """
        Zero out first and last frequency channels 50.
        Zero out GSM channels: 95 - 125
        """
        self.I[0:50,:] = np.zeros([50,self.samples_T])
        self.I[95:125,:] = np.zeros([30,self.samples_T])
        self.I[260:532,:] = np.zeros([272, self.samples_T])
        #self.I[347:532,:] = np.zeros([185, self.samples_T])
        self.I[794:900,:] = np.zeros([106, self.samples_T])
        self.I[-50:,:] = np.zeros([50,self.samples_T])

    def get_profile(self):
        self.profile = self.I.sum(axis=0)
        self.norm_profile = self.profile/max(self.profile)
        return self.profile

    def get_peak(self):
        return max(self.I.sum(axis=0))

    def get_floor(self):
        return np.mean(self.I.sum(axis=0)[0:1000])

    def get_fwhm(self):

        peak = self.get_peak()
        floor = self.get_floor()

        return ((peak - floor)/2) + floor

    def get_on_pulse(self):
        print("Analysis for: ", self.file_name)
        fwhm = self.get_fwhm()
        profile = self.get_profile()
        print(np.shape(profile))
        idx1 = np.abs(profile-fwhm).argmin()
        print(fwhm)
        profile[idx1] = 0
        idx2 = np.abs(profile-fwhm).argmin()

        # check to make sure that idx2 does not also come out with a pulse start value
        if np.abs(idx2-idx1) <= 50:
            profile[idx2] = 0
            idx2 = np.abs(profile - fwhm).argmin()

        if idx1 < idx2:
            self.pulse_start = idx1
            self.pulse_stop = idx2
        else:
            self.pulse_start = idx2
            self.pulse_stop = idx1

        self.pulse_width = self.pulse_stop - self.pulse_start

        print("start: ", self.pulse_start)
        print("stop :", self.pulse_stop)
        print("width:", self.pulse_width)

def incoherent_dedisperse(data, pulsar_tag):
    """

    :param data: summed pulsar profile ie re and im has already been combined. shape is num_ch x samples_T
    :param pulsar_tag: last 4 digits of observation code obtained from the file_name
    :return:
    """

    pulsar = pulsars[pulsar_tag]
    dm = pulsar['dm']
    f2 = 1712 - (freq_resolution / 2)
    for i, freq in enumerate(frequencies):
        delay = 10**6*(dispersion_constant * dm * (1 / f2**2 - 1 / freq**2)) # us
        num_2_roll = int(np.round(delay / time_resolution)) # samples
        data[i, :] = np.roll(data[i, :], num_2_roll)

    return data

