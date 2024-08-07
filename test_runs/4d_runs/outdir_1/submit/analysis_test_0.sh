#!/bin/bash
#SBATCH --job-name=0_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=outdir/log_data_analysis/0_test_%j.log

#SBATCH --mem-per-cpu=2G



mpirun parallel_bilby_analysis outdir/data/test_data_dump.pickle --nlive 1024 --label test_0 --sampling-seed 518559 --outdir /home/urash/twouters/projects/jim_meets_harmonic/4d_runs/outdir/result
