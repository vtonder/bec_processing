 #!/bin/bash

echo getting intensity 

for j in {256,512}
  do
  for i in {2,4,8,16}
    do
    timeout 7200 mpiexec -n 32 python intensity.py 2210 -r msk -M $j -m 1 -n $i -l 1s -u skmax 
    if [ $? -eq 124 ]
      then
        echo timed out on M $j and  n $i for msk
        exit
    fi
  
    timeout 7200 mpiexec -n 32 python intensity.py 2210 -r vmsk -M $j -m 1 -n $i -l 1s -u skmax 
    if [ $? -eq 124 ]
      then
        echo timed out on M $j and  n $i for vmsk
        exit
    fi
  
    #python get_msk_rfi.py MSK 256 1 $i 
  done
done


# used to get all data for lower 4sigma and upper 4sigma
#for j in {8192,4096,2048,1024,512,256,128,64}
#  do
#  echo $j
#  timeout 10800 mpiexec -n 32 python intensity.py 2210 -r sk -M $j -l 4s -u 4s
#  if [ $? -eq 124 ]
#    then
#      echo timed out on M $j 
#      exit
#  fi
#done

# used to compute all the lower and upper thresholds data sets, note compute M=64 separately, increasing the timeout to 3h allowed for it to compute 
#for j in {8192,4096,2048,1024,512,256,128,64}
#  do
#  echo $j
#  for i in {0_5s,0s,1s,2s,2_5s,3s,4s}
#    do
#    echo $i
#    timeout 10800 mpiexec -n 32 python intensity.py 2210 -r sk -M $j -l $i  
#    if [ $? -eq 124 ]
#      then
#        echo timed out on M $j and lower threshold $i
#        exit
#    fi
#  done
#  
#  for i in {0_5s,0s,1s,2s,2_5s,3s,4s,skmax}
#    do
#    timeout 10800 mpiexec -n 32 python intensity.py 2210 -r sk -M $j -u $i  
#    if [ $? -eq 124 ]
#      then
#        echo timed out on M $j and upper threshold $i
#        exit
#    fi
#  done
#done
#
echo done
