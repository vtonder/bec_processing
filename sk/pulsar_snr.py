import matplotlib.pyplot as plt
import numpy as np
from constants import pulsars, num_ch, frequencies, dispersion_constant, freq_resolution, time_resolution
import re
from common import get_freq_ch

def get_profile(I):
    norm = np.where(I > 0, 1, 0).sum(axis=0)
    prof = I.sum(axis=0) / norm

    return prof

def get_peak(prof):
    return max(prof)

def get_floor(prof, floor_len = 1000):
    return np.mean(prof[0:floor_len])

def get_fwhm(peak, floor):
    return ((peak - floor) / 2) + floor

def get_on_pulse(prof, fwhm, min_width = 50):
    idx1 = np.abs(prof - fwhm).argmin()
    prof[idx1] = 0
    idx2 = np.abs(prof - fwhm).argmin()

    # check to make sure that idx2 does not also come out with a pulse start value
    # Know J0437 won't have a pulse width smaller than 50
    while np.abs(idx2 - idx1) <= min_width:
        prof[idx2] = 0
        idx2 = np.abs(prof - fwhm).argmin()

    if idx1 < idx2:
        pulse_start = idx1
        pulse_stop = idx2
    else:
        pulse_start = idx2
        pulse_stop = idx1

    pulse_width = pulse_stop - pulse_start

    print("FWHM : ", fwhm)
    print("start: ", pulse_start)
    print("stop : ", pulse_stop)
    print("width: ", pulse_width)

    return pulse_start, pulse_stop, pulse_width

def snr_toa_un(profile, pulse_start, pulse_width):
    snr = 0
    m = np.mean(profile[0:pulse_width])
    s = np.std(profile[0:pulse_width])
    for i in np.arange(pulse_width):
        snr = snr + (profile[pulse_start + i] - m)
    snr = snr / (s * np.sqrt(pulse_width))
    toa_un = pulse_width * time_resolution / snr

    print("SNR         : ", snr)
    print("TOA un [us] : ", toa_un)

    return snr, toa_un

def calc_var(I):
    std = []
    for i in np.arange(num_ch):
        if not any(I[i,:]):
            continue
        profile = I[i,:] / np.mean(I[i,:])
        std.append(np.sqrt(np.var(profile[0:1000])))
    return std

def apply_sarao_mask(I):
    """
    Zero out first and last frequency channels 50.
    Zero out GSM channels: 95 - 125
    """
    samples_T = I.shape[1]
    I[0:50, :] = np.zeros([50, samples_T])
    I[95:126, :] = np.zeros([31, samples_T]) # GSM
    #I[260:532,:] = np.zeros([272, samples_T]) # includes aircraft and GNSS
    I[347:533,:] = np.zeros([186, samples_T]) # excludes aircraft only GNSS
    I[794:901,:] = np.zeros([107, samples_T]) # GNSS except Iridium
    #I[794:924,:] = np.zeros([130, samples_T]) # GNSS and Iridium
    I[-50:, :] = np.zeros([50, samples_T])

    return I

def apply_dspsr_mask(I):
    """
    From Andrew Jameson, he uses DSPSR to zap the following:
    855.5820313 - 856.4179688 ?
    934.9960938 - 960
    1087.136719 - 1093.824219
    1240 - 1250
    1525 - 1529
    1529.347656 - 1559
    1597.058594 - 1599
    1617.957031 - 1628
    :return:
    """
    samples_T = I.shape[1]
    zap_freq = [[934.9960938, 960], [1087.136719, 1093.824219], [1240, 1250], [1525, 1529], [1529.347656, 1559], [1597.058594,1599], [1617.957031, 1628]]
    I[0:50,:] = np.zeros([50, samples_T])
    I[-50:,:] = np.zeros([50, samples_T])
    for zf in zap_freq:
        start = get_freq_ch(zf[0])
        end = get_freq_ch(zf[1])
        I[start:end,:] = np.zeros([end-start, samples_T])

    return I

def apply_clipped_mask(I):
    samples_T = I.shape[1]
    I[95:126, :] = np.zeros([31, samples_T])

    return I

def apply_DME_mask(I):
    samples_T = I.shape[1]
    I[205:342, :] = np.zeros([137, samples_T])

    return I

def apply_GNSS_low_mask(I):
    samples_T = I.shape[1]
    I[347:533, :] = np.zeros([186, samples_T])

    return I

def apply_GNSS_high_mask(I):
    samples_T = I.shape[1]
    I[794:901, :] = np.zeros([107, samples_T])

    return I

def apply_iridium_mask(I):
    samples_T = I.shape[1]
    I[911:923, :] = np.zeros([12, samples_T])

    return I

regex = re.compile(r'\d+')
class PI:
    def __init__(self, dir, file_name, sf = None, initialise=True):
        """
        :param dir: location of data
        :param file_name: folded pulsar intensity file. freq channels X pulse num samples
        :param sf: summed flags
        :param initialise: compute snr and toa_uncertainty
        """

        self.file_name = file_name
        self.I = np.load(dir + file_name)
        self.num_pol = 2 # H and V polarisation

        # parse what the numbers in the file name means
        nums_str = regex.findall(file_name)
        nums_int = [int(i) for i in nums_str]
        self.tag = nums_str[-2]
        self.num_pulses = nums_int[-1]
        self.pulsar = pulsars[self.tag]
        self.samples_T = int(np.floor(self.pulsar['samples_T']))

        if sf:
            self.sf = np.load(dir + sf)
            self.rfi = 100*(np.float32(self.sf).sum(axis=1) / (self.num_pol*int(self.samples_T)*self.num_pulses))
        else:
            self.sf = None

        self.profile = get_profile(self.I)
        self.norm_profile = self.profile / max(self.profile)

        self.pulse_start = 0
        self.pulse_stop = 0
        self.pulse_width = 0
        self.std = []
        self.snr = 0
        self.toa_un = 0

        if initialise:
            #self.I = apply_sarao_mask(self.I)
            self.profile = get_profile(self.I)
            self.norm_profile = self.profile / max(self.profile)
            #self.pulse_start, self.pulse_stop, self.pulse_width, self.snr, self.toa_un = self.compute()
            self.compute()

    def compute(self):
        profile = get_profile(self.I)
        peak = get_peak(profile)
        floor = get_floor(profile)
        fwhm = get_fwhm(peak, floor)

        pulse_start, pulse_stop, pulse_width = get_on_pulse(profile, fwhm)
        snr, toa_un = snr_toa_un(profile, pulse_start, pulse_width)

        self.pulse_start, self.pulse_stop, self.pulse_width, self.snr, self.toa_un = pulse_start, pulse_stop, pulse_width, snr, toa_un

        #return pulse_start, pulse_stop, pulse_width, snr, toa_un
