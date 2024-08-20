#!/bin/bash
#SBATCH --job-name=0_bbh_2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=outdir/out_bbh_2/log_data_analysis/0_bbh_2_%j.log

#SBATCH --mem-per-cpu=2G



mpirun parallel_bilby_analysis outdir/out_bbh_2/data/bbh_2_data_dump.pickle --nlive 2000 --nact 40 --naccept 80 --label bbh_2_0 --sampling-seed 631086 --outdir /home/urash/twouters/projects/jim_meets_harmonic/batch_4d_runs/outdir/out_bbh_2/result
