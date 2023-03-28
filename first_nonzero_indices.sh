#!/bin/bash

files=$(ls /net/com08/data6/vereese/*.h5)

for file in $files
do
  python first_nonzero_indices.py $file
done
