export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

OUTDIR=/home/urash/twouters/projects/jim_meets_harmonic/11d_runs/outdir

N=1
for i in $(seq 0 $((N-1)))
do
    FULL_LABEL=bbh_$i
    FULL_OUTDIR=$OUTDIR/out_$FULL_LABEL
    mpiexec -np 16 parallel_bilby_analysis $FULL_OUTDIR/data/${FULL_LABEL}_data_dump.pickle \
        --label $FULL_LABEL \
        --sampling-seed 42 \
        --outdir $FULL_OUTDIR \
        --check-point-deltaT 1800 \
        --nlive 1024 \
        --naccept 60 
done