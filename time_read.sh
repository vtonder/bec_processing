#!/bin/bash

for i in {5,10,15,20,25,30,35,40,45,50}
  do
    mpiexec -n $i python time_reading_mpi.py 2210
done
