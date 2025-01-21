 #!/bin/bash

echo getting intensity 

# example of running power threshold rfi mitigation technique, default std is set to 14
# timeout 7200 mpiexec -n 32 python intensity.py 2210 -r pt 

# examples of running MSK and VSMK
#for j in {256,512}
#  do
#  for i in {2,4,8,16}
#    do
#    timeout 7200 mpiexec -n 32 python intensity.py 2210 -r msk -M $j -m 1 -n $i -l 1s -u skmax 
#    if [ $? -eq 124 ]
#      then
#        echo timed out on M $j and  n $i for msk
#        exit
#    fi
#  
#    timeout 7200 mpiexec -n 32 python intensity.py 2210 -r vmsk -M $j -m 1 -n $i -l 1s -u skmax 
#    if [ $? -eq 124 ]
#      then
#        echo timed out on M $j and  n $i for vmsk
#        exit
#    fi
#  
#    #python get_msk_rfi.py MSK 256 1 $i 
#  done
#done


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

# used to compute all the lower and upper thresholds data sets, note increased the timeout to 3h to allow M=64 to compute 
for j in {8192,4096,2048,1024,512,256,128,64}
  do
  echo $j
  for i in {0_5s,0s,1s,2s,2_5s,3s,4s,skmin}
    do
    echo $i
    timeout 10800 mpiexec -n 4 python intensity_1pol.py 1064 -p '/home/vereese/' -x y -r sk -M $j -l $i  
    #timeout 10800 mpiexec -n 4 python intensity.py 1064 -p '/home/vereese/' -r sk -M $j -l $i  
    if [ $? -eq 124 ]
      then
        echo timed out on M $j and lower threshold $i
        exit
    fi
  done
  
  for i in {0_5s,0s,1s,2s,2_5s,3s,4s,skmax}
    do
    echo $i
    #timeout 10800 mpiexec -n 4 python intensity.py 1064 -p '/home/vereese/' -r sk -M $j -u $i  
    timeout 10800 mpiexec -n 4 python intensity_1pol.py 1064 -p '/home/vereese/' -x y -r sk -M $j -u $i  
    if [ $? -eq 124 ]
      then
        echo timed out on M $j and upper threshold $i
        exit
    fi
  done

  #for i in {0_5s,0s,1s,2s,2_5s,3s,4s,skmin}
  #  do
  #  echo $i
  #  timeout 10800 mpiexec -n 32 python intensity.py 2210 -r sk -M $j -l $i  
  #  if [ $? -eq 124 ]
  #    then
  #      echo timed out on M $j and lower threshold $i
  #      exit
  #  fi
  #done
  
  #for i in {0_5s,0s,1s,2s,2_5s,3s,4s,skmax}
  #  do
  #  echo $i
  #  timeout 10800 mpiexec -n 32 python intensity.py 2210 -r sk -M $j -u $i  
  #  if [ $? -eq 124 ]
  #    then
  #      echo timed out on M $j and upper threshold $i
  #      exit
  #  fi
  #done
done

echo done
