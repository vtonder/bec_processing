from matplotlib import pyplot as plt
from constants import a4_textheight, a4_textwidth, thesis_font

# This data was generated using the time_reading_mpi.py script

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

processing_time ={
  '5':95.52786914348601,
  '10':45.981049015522004,
  '15':31.585434335867564,
  '20':24.818814676404,
  '25':22.571849013328556,
  '30':23.0546128765742,
  '35':19.676805268696377,
  '40':20.241357430815693,
  '45':21.547988188531665,
  '50':21.39174386906624
}

num_proc = [int(i) for i in processing_time.keys()]

plt.figure(0)
#plt.plot(num_proc, list(processing_time.values()), linewidth=2)
#plt.semilogy(num_proc, list(processing_time.values()), base = 2, linewidth=2)
plt.loglog(num_proc, list(processing_time.values()), base = 2, linewidth=2)
plt.xlim(num_proc[0], num_proc[-1])
plt.grid()
plt.xlabel("Number of processors")
plt.ylabel("$\overline{t_r}$ \,\, [s]")
plt.savefig('/home/vereese/Documents/PhD/Thesis/Figures/proctime.pdf', bbox_inches='tight')
plt.show()
