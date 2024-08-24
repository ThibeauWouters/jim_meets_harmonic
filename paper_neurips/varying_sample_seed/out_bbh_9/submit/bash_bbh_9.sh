jid0=$(sbatch out_bbh_9/submit/analysis_bbh_9_0.sh)
squeue -u $USER -o "%u %.10j %.8A %.4C %.40E %R"
