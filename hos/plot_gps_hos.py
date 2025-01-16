import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
from constants import a4_textwidth, a4_textheight, thesis_font

# Setup fonts and sizes for publication, based on page dimensions in inches
textwidth =  a4_textwidth
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

DIRECTORY = '/home/vereese/git/phd_data/gps_hos/output_data/'
URSI_DIR = '/home/vereese/Documents/PhD/ThesisTemplate/Figures/'
NUM_BITS1 = str(1)
NUM_BITS2 = str(300)

py = 10.23
fft_len = 1024
py_res = 2 * py / fft_len
ca = 1.023
ca_res = 2 * ca / fft_len
py_freq = np.arange(-py, py, py_res)
ca_freq = np.arange(-ca, ca, ca_res)
def get_freq(code):
    if code.find('PY') != -1:
        w0 = py_freq[0]
        w1 = py_freq[-1]
    else:
        w0 = ca_freq[0]
        w1 = ca_freq[-1]
    return w0,w1

gps_data1 = {'I_PY':[], 'Q_CA':[],
           'I_PY_D':[], 'Q_CA_D':[],
           'I_CA_D':[], 'Q_L2CM_Dc':[]}

gps_data2 = {'I_PY':[], 'Q_CA':[],
           'I_PY_D':[], 'Q_CA_D':[],
           'I_CA_D':[], 'Q_L2CM_Dc':[]}

names = list(gps_data2.keys())
for code in gps_data2.keys():
    gps_data1[code] = np.load(DIRECTORY + code + '_' + NUM_BITS1 + 'bit_bispec.npy')
    gps_data2[code] = np.load(DIRECTORY + code + '_' + NUM_BITS2 + 'bit_bispec.npy')

#a = list(np.sum(np.abs(gps_data2['I_PY'] - gps_data2['I_PY_D']), axis=0))
#if not any(a): print("they the same")

for i in np.arange(6):
    code = names[int(i)]
    w0,w1 = get_freq(code)
    plt.figure(i+6)
    plt.imshow(np.abs(gps_data1[code]), aspect='auto', origin='lower', extent=[w0, w1, w0, w1])
    plt.xlabel("$f_1$ [MHz]")
    plt.ylabel("$f_2$ [MHz]")
    #plt.title(code+NUM_BITS1)
    plt.savefig(URSI_DIR+code+'_1bit.pdf', bbox_inches='tight')
    plt.figure(i)
    plt.imshow(np.abs(gps_data2[code]), aspect='auto', origin='lower', extent=[w0, w1, w0, w1])
    plt.xlabel("$f_1$ [MHz]")
    plt.ylabel("$f_2$ [MHz]")
    #plt.title(code+NUM_BITS2)
    plt.savefig(URSI_DIR+code+'_300bit.pdf', bbox_inches='tight')

plt.show()
