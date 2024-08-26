#! /bin/bash

# Call parallel_bilby_generation on all the injections without SLURM hassle

N=10
for i in $(seq 0 $((N-1)) )
do
    echo "========== parallel_bilby_generation injection " $i
    parallel_bilby_generation bbh_${i}.ini &> generation_log/generation_injection_${i}.err
done