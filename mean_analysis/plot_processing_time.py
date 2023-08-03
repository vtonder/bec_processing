from matplotlib import pyplot as plt

textwidth = 9.6 # 128.0 / 25.4 #
textheight = 7 # 96.0 / 25.4 # 7
plt.rc('font', size=12, family='STIXGeneral')
plt.rc('pdf', fonttype=42)
#plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('axes', titlesize=12, labelsize=12)
plt.rc(('xtick', 'ytick'), labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('lines', markersize=5)
plt.rc('figure', figsize=(0.9 * textwidth, 0.8 * textheight), facecolor='w')
plt.rc('mathtext', fontset='stix')

processing_time ={
  '5': 3034.3588042259216,
  '10':1574.2632755279542,
  '15':1047.1665592193604,
  '20':789.898957490921,
  '25':709.3646445274353,
  '30':672.2443092823029,
  '35':643.1528521537781,
  '40':576.8024142742157,
  '45':537.296351480484
}

num_proc = [int(i) for i in processing_time.keys()]

plt.figure(0)
plt.plot(num_proc, list(processing_time.values()))
plt.xlim(num_proc[0], num_proc[-1])
plt.grid()
plt.xlabel("Number of processors")
plt.ylabel("Total processing time (s)")
plt.savefig('/home/vereese/Documents/PhD/ThesisTemplate/Figures/proctime', bbox_inches='tight')
plt.show()
