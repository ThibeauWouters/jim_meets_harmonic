#!/bin/bash
#SBATCH --job-name=0_bbh_0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=outdir/out_bbh_0/log_data_analysis/0_bbh_0_%j.log

#SBATCH --mem-per-cpu=2G



mpirun parallel_bilby_analysis outdir/out_bbh_0/data/bbh_0_data_dump.pickle --label bbh_0_0 --sampling-seed 819236 --outdir /home/urash/twouters/projects/jim_meets_harmonic/batch_4d_runs/outdir/out_bbh_0/result
