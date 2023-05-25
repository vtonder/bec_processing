from matplotlib import pyplot as plt
import numpy as np
from constants import pulsars, frequencies

"""
Band pass of, 
1569: Vela : not flat
2210: J0437: flat
2762: J0536: flat
3330: J0737: flat
3883: J0742: flat ?
4511: J1644: not flat
"""

def plot_vars_pol(vars, pol, i = 0):
    """
    Plot real and imaginary variances of a polarisation
    :param vars: variances of all pulsar data
    :param pol: string of "X" or "Y" polarisation
    :param i: initial figure number
    :return:
    """

    for pulsar_tag, var in vars.items():
        plt.figure(i, figsize=[15,12])
        plt.plot(frequencies[1:], var[1:, 0])
        plt.xlim([frequencies[1], frequencies[-1]])
        plt.title(pulsars[pulsar_tag]['name'] + ": " + pol + "-pol Real variance")
        plt.xlabel("Frequency MHz")
        plt.grid()

        plt.figure(i + 1, figsize=[15,12])
        plt.plot(frequencies[1:], var[1:, 1])
        plt.xlim([frequencies[1], frequencies[-1]])
        plt.title(pulsars[pulsar_tag]['name'] + ": " + pol + "-pol Imag variance")
        plt.xlabel("Frequency MHz")
        plt.grid()

        i = i + 2

font = {'family': 'STIXGeneral',
        'size': 26}
plt.rc('font', **font)

dir = "/home/vereese/git/phd_data/mean_analysis/"
fx_name = "/var_0x_1024.npy"
fy_name = "/var_0y_1024.npy"
vars_x = {}
vars_y = {}

for pulsar_tag in pulsars.keys():
    vars_x.update({pulsar_tag:np.load(dir+pulsar_tag+fx_name)})
    vars_y.update({pulsar_tag:np.load(dir+pulsar_tag+fy_name)})

plot_vars_pol(vars_x, "X")
plot_vars_pol(vars_y, "Y", i = 12)

plt.show()
