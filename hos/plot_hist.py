import pickle
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats

with open('/home/vereese/git/phd_data/hist/hist_2762_0x_96', 'rb') as inputfile:
    data = Counter(pickle.load(inputfile))

values = np.asarray([float(x) for x in data.keys()])
occ = np.asarray([float(x) for x in data.values()])
#in1 = np.where(values==-128)
#values = np.delete(values,in1)

#occ = np.delete(occ,in1)
#in2 = np.where(values==127)
#values = np.delete(values,in2)
#occ = np.delete(occ,in2)
#print(values)
var = sum(values**2*occ)/(sum(occ)-1)
#print(var)
std =  np.sqrt(var)
#print(std)

x1 = np.arange(-128,128)
#x1 = np.arange(-126,127)
s = np.random.normal(0, 1, 1000)

pdf_s = [1/(std*np.sqrt(2*np.pi))*np.exp(-0.5*(x/std)**2) for x in x1]
fit = stats.norm.pdf(x1, 0, std)


plt.figure()
#plt.bar(values, max(pdf_s)*(occ/max(occ)))
plt.bar(values, occ/sum(occ))
#plt.plot(x1, stats.norm.pdf(x1, 0, std))
#plt.plot(x1,pdf_s, label='fit using ')
plt.plot(x1,fit, 'r',label='PDF fit')
plt.xlabel("bins (8 bit => range [-128,127])")
plt.ylabel("normalised occurrences")
plt.legend()
plt.grid()
plt.show()