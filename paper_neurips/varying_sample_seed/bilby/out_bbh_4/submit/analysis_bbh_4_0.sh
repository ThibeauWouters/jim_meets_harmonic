#!/bin/bash
#SBATCH --job-name=0_bbh_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=out_bbh_4/log_data_analysis/0_bbh_4_%j.log

#SBATCH --mem-per-cpu=2G



mpirun parallel_bilby_analysis out_bbh_4/data/bbh_4_data_dump.pickle --nact 40 --naccept 80 --label bbh_4_0 --sampling-seed 9527 --outdir /home/urash/twouters/projects/jim_meets_harmonic/paper_neurips/varying_sample_seed/out_bbh_4/result
