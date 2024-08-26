#!/bin/bash
#SBATCH --job-name=0_bbh_6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=out_bbh_6/log_data_analysis/0_bbh_6_%j.log

#SBATCH --mem-per-cpu=2G



mpirun parallel_bilby_analysis out_bbh_6/data/bbh_6_data_dump.pickle --nact 40 --naccept 80 --label bbh_6_0 --sampling-seed 8564 --outdir /home/urash/twouters/projects/jim_meets_harmonic/paper_neurips/varying_sample_seed/out_bbh_6/result
