 #!/bin/bash

echo getting sk
mpiexec -n 32 python ps_sk_mpi.py 1604643883_wide_tied_array_channelised_voltage_0x.h5 -m 512
mpiexec -n 32 python ps_sk_mpi.py 1604643883_wide_tied_array_channelised_voltage_0y.h5 -m 512
mpiexec -n 32 python ps_sk_mpi.py 1604643883_wide_tied_array_channelised_voltage_0x.h5 -m 1024
mpiexec -n 32 python ps_sk_mpi.py 1604643883_wide_tied_array_channelised_voltage_0y.h5 -m 1024
mpiexec -n 32 python ps_sk_mpi.py 1604643883_wide_tied_array_channelised_voltage_0x.h5 -m 2048
mpiexec -n 32 python ps_sk_mpi.py 1604643883_wide_tied_array_channelised_voltage_0y.h5 -m 2048
echo done
echo calling ps
python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0x.h5 #no mitigation
python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0y.h5 #no mitigation
python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0x.h5 -m 512
python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0y.h5 -m 512
python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0x.h5 -m 1024
python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0y.h5 -m 1024
python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0x.h5 -m 2048
python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0y.h5 -m 2048
echo done
