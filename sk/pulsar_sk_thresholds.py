import numpy as np
from matplotlib import pyplot as plt

#M = ["64"] #, "128", "256", "512", "1024", "2048", "4096", "8192"]
M = ["128", "256", "512", "1024", "2048", "4096", "8192"]

#DIR = "/home/vereese/data/phd_data/sk_analysis/2210.bac/"
#DIR = "/home/vereese/git/bec_processing/sk/2210.bac/"

x = np.arange(0, 2, 0.01)

ch1 = np.arange(50,81)
ch2 = np.arange(126,200)
ch3 = np.arange(250,265)
ch4 = np.arange(532,794)
ch5 = np.arange(924,974)
clean_chs = np.concatenate([ch1,ch2,ch3,ch4,ch5])

for m in M:
    p4 = np.load("pdf_M"+str(m)+".npy")
    sk = np.load("sk_z_M" + str(m) + "_m1_n1_2210_0x.npy")
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

