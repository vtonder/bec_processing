import numpy as np
import h5py
from constants import start_indices

file_names = [
 '1604641064_wide_tied_array_channelised_voltage_0x.h5',
 '1604641064_wide_tied_array_channelised_voltage_0y.h5',
 '1604641234_wide_tied_array_channelised_voltage_0x.h5',
 '1604641234_wide_tied_array_channelised_voltage_0y.h5',
 '1604641569_wide_tied_array_channelised_voltage_0x.h5',
 '1604641569_wide_tied_array_channelised_voltage_0y.h5',
 '1604642210_wide_tied_array_channelised_voltage_0x.h5',
 '1604642210_wide_tied_array_channelised_voltage_0y.h5',
 '1604642762_wide_tied_array_channelised_voltage_0x.h5',
 '1604642762_wide_tied_array_channelised_voltage_0y.h5',
 '1604643330_wide_tied_array_channelised_voltage_0x.h5',
 '1604643330_wide_tied_array_channelised_voltage_0y.h5',
 '1604643883_wide_tied_array_channelised_voltage_0x.h5',
 '1604643883_wide_tied_array_channelised_voltage_0y.h5',
 '1604644511_wide_tied_array_channelised_voltage_0x.h5',
 '1604644511_wide_tied_array_channelised_voltage_0y.h5'
]

offsets = {}
for fn in file_names:
    offsets.update({fn:0})

number_observations = int(len(file_names)/2)

for i in np.arange(number_observations):
    print(file_names[2 * i])
    xf = h5py.File('/net/com08/data6/vereese/'+file_names[2*i], 'r')
    yf = h5py.File('/net/com08/data6/vereese/'+file_names[2*i + 1], 'r')
    xt1 = xf['Data/timestamps'][start_indices[file_names[2*i]]]
    yt1 = yf['Data/timestamps'][start_indices[file_names[2*i + 1]]]
    x_offset = int(0)
    y_offset = int(0)
    if xt1 < yt1:
        print("xt1 < yt1")
        print("xt1:", xt1)
        print("yt1:", yt1)
        x_offset = int((yt1 - xt1) / 2048)
    elif yt1 < xt1:
        print("yt1 < xt1")
        print("xt1:", xt1)
        print("yt1:", yt1)
        y_offset = int((xt1 - yt1) / 2048)

    offsets[file_names[2*i]] = x_offset
    offsets[file_names[2*i + 1]] = y_offset
    if xf['Data/timestamps'][x_offset+start_indices[file_names[2*i]]] == yf['Data/timestamps'][y_offset + start_indices[file_names[2*i + 1]]]:
        print("offset correctly calculated")
    else:
        print("offset WRONG!")

np.save("xy_offsets", offsets)
