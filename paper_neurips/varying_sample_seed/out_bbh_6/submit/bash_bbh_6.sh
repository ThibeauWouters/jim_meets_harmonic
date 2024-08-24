jid0=$(sbatch out_bbh_6/submit/analysis_bbh_6_0.sh)
squeue -u $USER -o "%u %.10j %.8A %.4C %.40E %R"
