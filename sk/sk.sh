 #!/bin/bash

echo getting sk
#timeout 7200 mpiexec -n 32 python sk_intensity.py 2210 -M 4096 -m 1 -n 16 
#timeout 7200 mpiexec -n 32 python s1_s2_mpi.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 512
#if [ $? -eq 124 ]
#  then
#    echo timed out
#    exit
#fi
#
#timeout 7200 mpiexec -n 32 python s1_s2_mpi.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 512
#if [ $? -eq 124 ]
#  then
#    echo timed out
#    exit
#fi

#timeout 7200 python sk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 2048 -m 1 -n 1 -f sk -d g
#if [ $? -eq 124 ]
#  then
#    echo timed out
#    exit
#fi
#
#timeout 7200 python sk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 2048 -m 1 -n 1 -f sk -d g
#if [ $? -eq 124 ]
#  then
#    echo timed out
#    exit
#fi

#timeout 7200 mpiexec -n 32 python sk_intensity.py 2210 -M 8192
#if [ $? -eq 124 ]
#  then
#    echo timed out
#    exit
#fi


#for i in {64,128,256,512,1024,2048,4096,8192}
for i in {2,4,8,16}
  do
  timeout 7200 python sk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 512 -m 1 -n $i -f msk 
  if [ $? -eq 124 ]
    then
      echo timed out
      exit
  fi

  timeout 7200 python sk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 512 -m 1 -n $i -f msk 
  if [ $? -eq 124 ]
    then
      echo timed out
      exit
  fi

  #python get_msk_rfi.py MSK 256 1 $i 
done

#for i in {8,4,16}
#  do
#  python get_msk_rfi.py MSK 256 1 $i 
#done

#mpiexec -n 32 python sk_intensity.py 2210 -m 2048
#
#for i in {64} #,128,256,512,1024,2048,4096,8192,16384}
#  do 
 # echo $i
  #mpiexec -n 32 python s1_s2_mpi.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -m $i 
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 64 -m 4 -n 4
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 64 -m 4 -n 4
#mpiexec -n 32 python sk_intensity.py 2210 -M 64 -m 4 -n 4
#mpiexec -n 32 python sk_intensity.py 2210 -M 64 -m 2 -n 4
#mpiexec -n 32 python sk_intensity.py 2210 -M 64 -m 4 -n 2

#mpiexec -n 32 python s1_s2_mpi.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -m 256 
#mpiexec -n 32 python s1_s2_mpi.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -m 256 

#python msk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 64 -m 2 -n 8
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 64 -m 2 -n 8
##mpiexec -n 32 python sk_integrated_intensity.py 2210 -m 64 
#mpiexec -n 32 python sk_integrated_intensity.py 2210 -m 128 
#mpiexec -n 32 python sk_integrated_intensity.py 2210 -m 256
#mpiexec -n 32 python sk_integrated_intensity.py 2210 -m 512
#
#python get_msk_rfi.py VMSK 4096 1 2
#python get_msk_rfi.py VMSK 4096 1 16 
#python get_msk_rfi.py VMSK 4096 1 4
#python get_msk_rfi.py VMSK 4096 1 8
#
#python get_msk_rfi.py MSK 4096 1 2
#python get_msk_rfi.py MSK 4096 1 16 
#python get_msk_rfi.py MSK 4096 1 4
#python get_msk_rfi.py MSK 4096 1 8



#mpiexec -n 32 python sk_intensity.py 2210 -M 64 -m 1 -n 8
#mpiexec -n 32 python vmsk_intensity.py 2210 -M 64 -m 1 -n 8
#
#mpiexec -n 32 python sk_intensity.py 2210 -M 64 -m 1 -n 2
#mpiexec -n 32 python vmsk_intensity.py 2210 -M 64 -m 1 -n 2
#
#mpiexec -n 32 python sk_intensity.py 2210 -M 64 -m 1 -n 4
#mpiexec -n 32 python vmsk_intensity.py 2210 -M 64 -m 1 -n 4
#
#mpiexec -n 32 python sk_intensity.py 2210 -M 64 -m 1 -n 16
#mpiexec -n 32 python vmsk_intensity.py 2210 -M 64 -m 1 -n 16


#python msk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 128 -m 1 -n 2
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 128 -m 1 -n 2
#mpiexec -n 32 python sk_intensity.py 2210 -M 128 -m 1 -n 2
#
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 128 -m 1 -n 4
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 128 -m 1 -n 4
#mpiexec -n 32 python sk_intensity.py 2210 -M 128 -m 1 -n 4
#
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 128 -m 1 -n 8
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 128 -m 1 -n 8
#mpiexec -n 32 python sk_intensity.py 2210 -M 128 -m 1 -n 8
#
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 128 -m 1 -n 16
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 128 -m 1 -n 16
#mpiexec -n 32 python sk_intensity.py 2210 -M 128 -m 1 -n 16
#
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 256 -m 1 -n 2
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 256 -m 1 -n 2
#mpiexec -n 32 python sk_intensity.py 2210 -M 256 -m 1 -n 2
#
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 256 -m 1 -n 4
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 256 -m 1 -n 4
#mpiexec -n 32 python sk_intensity.py 2210 -M 256 -m 1 -n 4
#
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 256 -m 1 -n 8
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 256 -m 1 -n 8
#mpiexec -n 32 python sk_intensity.py 2210 -M 256 -m 1 -n 8
#
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 256 -m 1 -n 16
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 256 -m 1 -n 16
#mpiexec -n 32 python sk_intensity.py 2210 -M 256 -m 1 -n 16

#for i in {4,2,16}
#  do 
#        echo $i  
#        python msk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M 4096 -n $i -m 1
#        python msk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M 4096 -n $i -m 1
#        mpiexec -n 32 python sk_intensity.py 2210 -M 4096 -n $i -m 1
#done

#python msk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M $i -m 2 -n 4
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M $i -m 2 -n 4

#python msk.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -M $i -m 4 -n 2
#python msk.py 1604642210_wide_tied_array_channelised_voltage_0y.h5 -M $i -m 4 -n 2
#mpiexec -n 32 python sk_intensity.py 2210 -M 64 -m 4 -n 2

#mpiexec -n 32 python sk_mpi.py 1604641234_wide_tied_array_channelised_voltage_0x.h5
#mpiexec -n 32 python sk_mpi.py 1604641234_wide_tied_array_channelised_voltage_0y.h5
#mpiexec -n 32 python sk_integrated_intensity.py 2210 -m 1024
#mpiexec -n 32 python sk_integrated_intensity.py 2210 -m 2048
#mpiexec -n 32 python sk_integrated_intensity.py 2210 -m 4096
#mpiexec -n 32 python sk_integrated_intensity.py 2210 -m 8192
#mpiexec -n 32 python sk_integrated_intensity.py 2210 -m 16384 
#mpiexec -n 32 python ps_sk_mpi.py 1604643883_wide_tied_array_channelised_voltage_0x.h5 -m 512
#mpiexec -n 32 python ps_sk_mpi.py 1604643883_wide_tied_array_channelised_voltage_0y.h5 -m 512
#mpiexec -n 32 python sk_mpi.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -m 16384
#mpiexec -n 32 python sk_mpi.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -m 8192
#mpiexec -n 32 python sk_mpi.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -m 4096
#mpiexec -n 32 python sk_mpi.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -m 256
#mpiexec -n 32 python sk_mpi.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -m 128
#mpiexec -n 32 python sk_mpi.py 1604642210_wide_tied_array_channelised_voltage_0x.h5 -m 64

#mpiexec -n 32 python ps_sk_mpi.py 1604643883_wide_tied_array_channelised_voltage_0x.h5 -m 2048
#mpiexec -n 32 python ps_sk_mpi.py 1604643883_wide_tied_array_channelised_voltage_0y.h5 -m 2048
echo done
#echo calling ps
#mpiexec -n 16 python sk_intensity.py 1569 -m 1024
#python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0y.h5 #no mitigation
#python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0x.h5 -m 512
#python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0y.h5 -m 512
#python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0x.h5 -m 1024
#python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0y.h5 -m 1024
#python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0x.h5 -m 2048
#python ps_sk.py 1604643883_wide_tied_array_channelised_voltage_0y.h5 -m 2048
#echo done
