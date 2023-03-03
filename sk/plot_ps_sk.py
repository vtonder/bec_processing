import numpy as np
import sys
sys.path.append('../')
from constants import *
from matplotlib import pyplot as plt

direc = './'
code = '2762'
pulsar = pulsars[code] 

spx = {'ps_'+code+'_0x.npy':[]}#, 'ps_M512_'+code+'_0x.npy':[], 'ps_M1024_'+code+'_0x.npy':[], 'ps_M2048_'+code+'_0x.npy':[], 'ps_M10240_'+code+'_0x.npy':[]} 
spy = {'ps_'+code+'_0y.npy':[]}#, 'ps_M512_'+code+'_0y.npy':[], 'ps_M1024_'+code+'_0y.npy':[], 'ps_M2048_'+code+'_0y.npy':[], 'ps_M10240_'+code+'_0y.npy':[]} 
sp = {'ps_'+code:[]}#, 'ps_M512_'+code:[], 'ps_M1024_'+code:[], 'ps_M2048_'+code:[], 'ps_M10240_'+code:[]}

px = {'ps_'+code+'_0x.npy':[]}#, 'ps_M512_'+code+'_0x.npy':[], 'ps_M1024_'+code+'_0x.npy':[], 'ps_M2048_'+code+'_0x.npy':[], 'ps_M10240_'+code+'_0x.npy':[]} 
px_max = {'ps_'+code+'_0x.npy':0}#, 'ps_M512_'+code+'_0x.npy':0, 'ps_M1024_'+code+'_0x.npy':0, 'ps_M2048_'+code+'_0x.npy':0, 'ps_M10240_'+code+'_0x.npy':0} 
py = {'ps_'+code+'_0y.npy':[]}#, 'ps_M512_'+code+'_0y.npy':[], 'ps_M1024_'+code+'_0y.npy':[], 'ps_M2048_'+code+'_0y.npy':[], 'ps_M10240_'+code+'_0y.npy':[]} 
py_max = {'ps_'+code+'_0x.npy':0}#, 'ps_M512_'+code+'_0x.npy':0, 'ps_M1024_'+code+'_0x.npy':0, 'ps_M2048_'+code+'_0x.npy':0, 'ps_M10240_'+code+'_0y.npy':0} 
p = {'ps_'+code:[]}#, 'ps_M512_'+code:[], 'ps_M1024_'+code:[], 'ps_M2048_'+code:[], 'ps_M10240_'+code:[]} 

summed_flagsx = {}#{'summed_flags512_'+code+'_0x.npy':[], 'summed_flags1024_'+code+'_0x.npy':[], 'summed_flags2048_'+code+'_0x.npy':[], 'summed_flags10240_'+code+'_0x.npy':[]}
summed_flagsy = {}#{'summed_flags512_'+code+'_0y.npy':[], 'summed_flags1024_'+code+'_0y.npy':[], 'summed_flags2048_'+code+'_0y.npy':[], 'summed_flags10240_'+code+'_0y.npy':[]}
summed_flags = {}#{'summed_flags512_'+code:[], 'summed_flags1024_'+code:[], 'summed_flags2048_'+code:[], 'summed_flags10240_'+code:[]}

sk ={}# {'SK_flags512_'+code+'_0x.npy':[], 'SK_flags1024_'+code+'_0x.npy':[], 'SK_flags2048_'+code+'_0x.npy':[], 'SK_flags10240_'+code+'_0x.npy':[]}


for fn, data in summed_flagsx.items():
    summed_flagsx[fn] = np.load(direc+fn)

for fn, data in summed_flagsy.items():
    summed_flagsy[fn] = np.load(direc+fn)

for fn, data in sk.items():
    sk[fn] = np.load(direc+fn)

for fn, data in spx.items():
    spx[fn] = np.load(direc+fn)

for fn, data in spy.items():
    spy[fn] = np.load(direc+fn)

f2 = 1712 - (freq_resolution / 2)
j, k = 0, 0
dm = pulsar['dm']

for i, freq in enumerate(frequencies):
    delay = 10 ** 6 * (dispersion_constant * dm * (1/(f2**2) - 1 / (freq ** 2)))  # us
    num_2_roll = int(np.round(delay / time_resolution))

    for fn, data in spx.items(): 
        spx[fn][i,:] = np.roll(spx[fn][i,:], num_2_roll)
    for fn, data in summed_flagsx.items():
       summed_flagsx[fn][i,:] = np.roll(summed_flagsx[fn][i,:], num_2_roll)

    for fn, data in spy.items(): 
        spy[fn][i,:] = np.roll(spy[fn][i,:], num_2_roll)
    for fn, data in summed_flagsy.items():
       summed_flagsy[fn][i,:] = np.roll(summed_flagsy[fn][i,:], num_2_roll)


mid_T = int(pulsar['samples_T']/2) 

for fn, data in px.items():
    px[fn] = np.sum(spx[fn], axis=0)
    px_max[fn] = px[fn].argmax()
    px[fn] = np.roll(px[fn], mid_T-px_max[fn])
    spx[fn] = np.roll(spx[fn], mid_T-px_max[fn])
for fn, data in summed_flagsx.items():
    summed_flagsx[fn] = np.roll(summed_flagsx[fn], mid_T-px_max['ps_M'+fn[12:]])

for fn, data in py.items():
    py[fn] = np.sum(spy[fn], axis=0)
    py_max[fn] = py[fn].argmax()
    py[fn] = np.roll(py[fn], mid_T-py_max[fn])
    spy[fn] = np.roll(spy[fn], mid_T-py_max[fn])
for fn, data in summed_flagsy.items():
    summed_flagsy[fn] = np.roll(summed_flagsy[fn], mid_T-py_max['ps_M'+fn[12:]])

for fn, data in sp.items():
    x = fn+'_0x.npy'
    y = fn+'_0y.npy'

    sp[fn] = spx[x]**2 + spy[y]**2
    p[fn] = px[x]**2 + py[y]**2

for fn, data in summed_flags.items():
    x = fn+'_0x.npy'
    y = fn+'_0y.npy'
    summed_flags[fn] = summed_flagsx[x]**2 + summed_flagsy[y]**2

# remove noise floor
'''for fn, data in p.items():
    p1 = np.mean(data[0:33000])
    p2 = np.mean(data[45000:])
    mean_p = np.mean([p1, p2])
    p[fn] = p[fn]-mean_p'''

plt.figure(0)
for fn, data in p.items():
    plt.plot(data/max(data), label=fn)
plt.xlabel('samples')
plt.title('normalised')
plt.legend()
plt.grid()
#plt.savefig(direc+'vela_normalised', bbox_inches='tight')

plt.figure(1)
for fn, data in p.items():
    plt.plot(data, label=fn)
plt.xlabel('samples')
plt.title('non normalised')
plt.legend()
plt.grid()
#plt.savefig(direc+'vela_non_normalised', bbox_inches='tight')

i = 2
mini = min(sp['ps_'+code].flatten())
maxi = max(sp['ps_'+code].flatten())/8 #choose max of pulsar, 600 is known rfi free channel 

for fn, data in sp.items(): 
    plt.figure(i)
    plt.imshow(data, aspect='auto', origin='lower', vmin=mini, vmax=maxi)
    plt.title(fn)
    #plt.savefig(direc+fn, bbox_inches='tight')
    i = i + 1

'''for fn, data in summed_flags.items():
    plt.figure(i)
    plt.imshow(data, aspect='auto', origin='lower')
    plt.title(fn)
    #plt.savefig(direc+fn[:-4], bbox_inches='tight')
    i = i + 1

for fn, data in sk.items():
    plt.figure(i)
    plt.imshow(data, aspect='auto', origin='lower')
    plt.title(fn[:-4])
    #plt.savefig(direc+fn[:-4], bbox_inches='tight')
    i = i + 1
'''
plt.show()
