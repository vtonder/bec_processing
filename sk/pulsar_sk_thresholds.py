import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
from constants import a4_textwidth, a4_textheight, thesis_font, frequencies

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

M = ["64", "128", "256", "512", "1024", "2048", "4096", "8192"]

#DIR = "/home/vereese/data/phd_data/sk_analysis/2210.bac/"
#DIR = "/home/vereese/git/bec_processing/sk/2210.bac/"

x = np.arange(0, 2, 0.01)

ch1 = np.arange(50,81)
ch2 = np.arange(126,200)
ch3 = np.arange(250,265)
ch4 = np.arange(532,794)
ch5 = np.arange(924,974)
clean_chs = np.concatenate([ch1,ch2,ch3,ch4,ch5])

"""
# A plot to show clean channels
fig, ax = plt.subplots()
band_x = np.load("/home/vereese/git/phd_data/mean_analysis/2210/var_0x_1024.npy")
band_y = np.load("/home/vereese/git/phd_data/mean_analysis/2210/var_0y_1024.npy")
band = np.sqrt(band_x)
ax.plot(frequencies, band)
ax.set_xlim([frequencies[0], frequencies[-1]])
ax.set_ylim([0, 28])
ax.set_xlabel("frequency [MHz]")
ax.set_ylabel("$\sigma_X$")
plt.axvspan(frequencies[int(ch1[0])], frequencies[int(ch1[-1])], color='blue', alpha=0.5)
plt.axvspan(frequencies[int(ch2[0])], frequencies[int(ch2[-1])], color='blue', alpha=0.5)
plt.axvspan(frequencies[int(ch3[0])], frequencies[int(ch3[-1])], color='blue', alpha=0.5)
plt.axvspan(frequencies[int(ch4[0])], frequencies[int(ch4[-1])], color='blue', alpha=0.5)
plt.axvspan(frequencies[int(ch5[0])], frequencies[int(ch5[-1])], color='blue', alpha=0.5)
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/clean_ch.pdf', transparent=True, bbox_inches='tight')
plt.show()

# A plot for SK max and SK min section in SK chapter of thesis
m = 256
# pearson type IV PDF produced by sk/sk_thresholds.py script
p4 = np.load("pdf_M" + str(m) + ".npy")
sk = np.load("sk_z_M" + str(m) + "_m1_n1_2210_0y.npy")
fig, ax = plt.subplots()
ax.semilogy(x, p4, label="Pearson type IV PDF", linewidth=2)
x_max = 678
y_max = 137 
ax.hist(sk[y_max,:], 1000, density = "True", stacked = "True", log = "True", label="SK histogram")
ax.set_xlim([0, 12])
ax.set_ylim([10**-4, 10])
ax.vlines(x=0.25, ymin = 10**-4, ymax = 10, color='m', label="minimum", linewidth=2)
ax.vlines(x=3.8, ymin = 10**-4, ymax = 10, color='g', label="median", linewidth=2)
ax.vlines(x=10.77, ymin = 10**-4, ymax = 10, color='r', label="maximum", linewidth=2)
plt.legend()
ax.set_ylabel("SK PDF")
ax.set_xlabel("SK values")
plt.savefig('/home/vereese/thesis_pics/sk_min_max_hist.pdf', transparent=True, bbox_inches='tight')
plt.show()
"""

for m in M:
    p4 = np.load("pdf_M"+str(m)+".npy")
    sk = np.load("sk_z_M" + str(m) + "_m1_n1_2210_0y.npy")
    max_sk = []
    min_sk = []

    for ch in clean_chs:
        mxs = max(sk[ch,:])
        if np.isnan(mxs):
            continue 
        max_sk.append(mxs)

    for ch in clean_chs:
        mins = min(sk[ch,:])
        if np.isnan(mins):
            continue 
        min_sk.append(mins)

    print("M        : ", m)
    print("max sk   : ", max(max_sk))
    print("freq ch  max sk : ", clean_chs[int(max_sk.index(max(max_sk)))])
    print("min sk   : ", min(min_sk))

    """#print("median sk:", np.median(max_sk))
    print("\n") 
    plt.figure(0)
    #plt.semilogy(x, p4)
    plt.hist(sk[420,:],1000,density="True",stacked="True",log="True")
    #plt.hist(sk[105,:],1000,density="True",stacked="True",log="True")
    plt.grid()
    plt.ylabel("SK PDF")
    #plt.ylim([10**-14,10**2])
    plt.xlabel("SK")
    plt.title("ch 420")
    #plt.title("clipped")

    plt.figure(1)
    #plt.semilogy(x, p4)
    plt.hist(sk[383,:],1000,density="True",stacked="True",log="True")
    #plt.hist(sk[150,:],1000,density="True",stacked="True",log="True")
    plt.grid()
    plt.ylabel("SK PDF")
    #plt.ylim([10**-14,10**2])
    plt.xlabel("SK")
    plt.title("GNSS ch 383")
    plt.show()
    #plt.title("clean data")


    plt.figure(2)
    plt.semilogy(x, p4)
    plt.hist(sk[280,:],1000,density="True",stacked="True",log="True")
    plt.grid()
    plt.ylabel("SK PDF")
    plt.ylim([10**-14,10**2])

    plt.xlabel("SK")
    plt.title("DME")

    plt.figure(3)
    plt.semilogy(x, p4)
    plt.hist(sk[419,:],1000,density="True",stacked="True",log="True")
    plt.grid()
    plt.ylabel("SK PDF")
    plt.ylim([10**-14,10**2])

    plt.xlabel("SK")
    plt.title("GNSS")

    plt.figure(4)
    plt.semilogy(x, p4)
    plt.hist(sk[492,:],1000,density="True",stacked="True",log="True")
    plt.grid()
    plt.ylabel("SK PDF")
    plt.ylim([10**-14,10**2])

    plt.xlabel("SK")
    plt.title("GNSS")

    plt.figure(5)
    plt.semilogy(x, p4)
    plt.hist(sk[620,:],1000,density="True",stacked="True",log="True")
    plt.grid()
    plt.ylabel("SK PDF")
    plt.ylim([10**-14,10**2])

    plt.xlabel("SK")
    plt.title("clean data")

    plt.figure(6)
    plt.semilogy(x, p4)
    plt.hist(sk[674,:],1000,density="True",stacked="True",log="True")
    plt.grid()
    plt.ylabel("SK PDF")
    plt.ylim([10**-14,10**2])

    plt.xlabel("SK")
    plt.title("HI")

    plt.figure(7)
    plt.semilogy(x, p4)
    plt.hist(sk[770,:],1000,density="True",stacked="True",log="True")
    plt.grid()
    plt.ylabel("SK PDF")
    plt.ylim([10**-14,10**2])

    plt.xlabel("SK")
    plt.title("clean")

    plt.figure(8)
    plt.semilogy(x, p4)
    plt.hist(sk[859,:],1000,density="True",stacked="True",log="True")
    plt.grid()
    plt.ylabel("SK PDF")
    plt.ylim([10**-14,10**2])

    plt.xlabel("SK")
    plt.title("GNSS")

    plt.show()"""

