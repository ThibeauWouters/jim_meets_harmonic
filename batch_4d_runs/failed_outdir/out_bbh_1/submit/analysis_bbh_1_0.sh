#!/bin/bash
#SBATCH --job-name=0_bbh_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=outdir/out_bbh_1/log_data_analysis/0_bbh_1_%j.log

#SBATCH --mem-per-cpu=2G



mpirun parallel_bilby_analysis outdir/out_bbh_1/data/bbh_1_data_dump.pickle --nact 20 --label bbh_1_0 --sampling-seed 370346 --outdir /home/urash/twouters/projects/jim_meets_harmonic/batch_4d_runs/outdir/out_bbh_1/result
