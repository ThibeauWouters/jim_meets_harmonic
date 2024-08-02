jid0=$(sbatch outdir/submit/analysis_test_0.sh)
squeue -u $USER -o "%u %.10j %.8A %.4C %.40E %R"
