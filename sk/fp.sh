 #!/bin/bash

echo getting sk
mpiexec -n 32 python flagged_power.py 2210 -m 64 
mpiexec -n 32 python flagged_power.py 2210 -m 128 
mpiexec -n 32 python flagged_power.py 2210 -m 256
mpiexec -n 32 python flagged_power.py 2210 -m 512
mpiexec -n 32 python flagged_power.py 2210 -m 1024
mpiexec -n 32 python flagged_power.py 2210 -m 2048
#mpiexec -n 32 python flagged_power.py 2210 -m 4096
mpiexec -n 32 python flagged_power.py 2210 -m 8192
mpiexec -n 32 python flagged_power.py 2210 -m 16384 
echo done

