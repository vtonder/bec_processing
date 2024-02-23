from matplotlib import pyplot as plt
# This script calculates answers as per the pulsar searching activity
# Note the delays are much more than the period of the pulsar
# Therefore higher frequency components from next pulse arrives when lower frequency components of first pulse
# gets to telescope

#Q1: calculate the delays between when higer and lower frequency components will arrive

delays = [] # will be in ms
f1=[300, 310, 325, 330, 350, 375] # MHz
f2=400 #MHz

dm=20 #parsec / cm^3
c=4.15e6

for f in f1:
    delays.append(c*dm*(1/(f**2) - 1/(f2**2)))

print(delays)

# i guess we expect some kind of hyperbolic activity
plt.figure()
plt.plot(f1,delays)
#plt.show()

fc = 350 #MHz
BW = 100 # MHz
ch = 4096
sampling=81.92 #us
DM = 20
pulsar_T = 3 #ms

freq_res = BW/ch
delta_t = 8.3e6 * DM * (freq_res/fc**3)

# To calculate at what DM the smearing delay will be 10% of the pulse period
DM=(pulsar_T*0.1)/(8.3e6)*(fc**3/freq_res)
print(DM)

# To calculate DM that equals smearing to be same as the sampling rate
DM=(sampling*1000)/(8.3e6)*(fc**3/freq_res)
''' Questions
- what does happen when we don't take DM into account?

'''

