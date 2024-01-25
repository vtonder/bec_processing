import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
from constants import time_resolution, thesis_font, a4_textwidth, a4_textheight
import argparse
import re

# Setup fonts and sizes for publication, based on page dimensions in inches
textwidth = a4_textwidth
textheight = a4_textheight
font_size = thesis_font
# groups are like plt.figure plt.legend etc
plt.rc('font', size=font_size, family='serif')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=font_size, labelsize=font_size)
plt.rc(('xtick', 'ytick'), labelsize=font_size)
plt.rc('legend', fontsize=font_size)
plt.rc('lines', markersize=5)
# The following should only be used for beamer
# plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
figheight = 0.65 * textwidth
plt.rc('mathtext', fontset='cm')
# to get this working needed to do: sudo apt install cm-super
plt.rc("text", usetex = True)
plt.rc("figure", figsize = (textwidth, figheight))

parser = argparse.ArgumentParser()
parser.add_argument("-d", dest = "dir", help = "directory where data is located. default location: /home/vereese/data/phd_data/sk_analysis/2210/", default = "/home/vereese/data/phd_data/sk_analysis/2210/")
parser.add_argument("-f", dest = "file", help = "SK flags file to plot. default: sk_xpol_flags_z_l1siguskmax_M256_m1_n1_2210_p45216.npy ", default = "sk_xpol_flags_z_l1siguskmax_M256_m1_n1_2210_p45216.npy")
parser.add_argument("-o", dest = "dirout", help = "output directory to save .pdf file to. default = /home/vereese/thesis_pics/", default = "/home/vereese/thesis_pics/")

args = parser.parse_args()
DIR = args.dir
DIR_OUT = args.dirout
file = args.file
data = np.load(DIR + file)

file_args = file.split("_")
regex = re.compile(r'\d+')
# the 5th _ in the file name gives _Mvalue_
M = int(regex.findall(file_args[5])[0])
# data has shape number frequency channels X number of SK windows
num_sk = data.shape[1]
obs_len = np.round(num_sk*M*time_resolution/10**6) # SK flags length in seconds
# second last substring is the observation tag
tag = file_args[-2]

fig, ax = plt.subplots()
ax.imshow(data, orign = "lower", aspect = "auto", extent = [856, 1712, 0, obs_len])
ax.set_xlabel("observation time [s]")
ax.set_ylabel("frequency [MHz]")
plt.savefig(DIR_OUT + 'sk_flags_' + tag + '_' + str(M) + '.pdf', bbox_inches='tight')