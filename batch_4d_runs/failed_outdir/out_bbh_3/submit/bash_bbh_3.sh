jid0=$(sbatch outdir/out_bbh_3/submit/analysis_bbh_3_0.sh)
squeue -u $USER -o "%u %.10j %.8A %.4C %.40E %R"
