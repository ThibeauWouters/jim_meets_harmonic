jid0=$(sbatch out_bbh_7/submit/analysis_bbh_7_0.sh)
squeue -u $USER -o "%u %.10j %.8A %.4C %.40E %R"
