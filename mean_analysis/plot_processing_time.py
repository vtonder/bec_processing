from matplotlib import pyplot as plt

font = {'family': 'STIXGeneral',
        'size': 26}
plt.rc('font', **font)

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

plt.figure(0, figsize=[15,12])
plt.plot(num_proc, list(processing_time.values()))
plt.xlim(num_proc[0], num_proc[-1])
plt.grid()
plt.xlabel("Number of processors")
plt.ylabel("Total processing time (s)")
plt.savefig('/home/vereese/Documents/PhD/CASPER2022/presentation/proctime', bbox_inches='tight')
plt.show()
