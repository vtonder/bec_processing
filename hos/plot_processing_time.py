from matplotlib import pyplot as plt
processing_time ={
  '5':2951.9122191639617,##/2919.0,
  '10':1477.788330548443,#/1459.0,
  '15':1003.9506793580949,#/973.0,
  '20':759.8343786997721,#/729.0,
  '25':626.7423422709107,#/583.0,
  '30':541.3413198925555,#/486.0,
  '35':482.35602218285203,#/417.0,
  '40':593.6757857976481,#/364.0
  '45':1745.7610048977658 #/364.0
}

num_proc = [int(i) for i in processing_time.keys()]

plt.figure(0)
plt.plot(num_proc, list(processing_time.values()))
plt.xlim(num_proc[0], num_proc[-1])
plt.grid()
plt.xlabel("number of processors")
plt.ylabel("processing time per chunk")
plt.show()
