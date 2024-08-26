jid0=$(sbatch out_bbh_4/submit/analysis_bbh_4_0.sh)
squeue -u $USER -o "%u %.10j %.8A %.4C %.40E %R"
