export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

OUTDIR=/home/urash/twouters/projects/jim_meets_harmonic/outdir
LABEL=test

mpiexec -np 16 parallel_bilby_analysis outdir/data/test_data_dump.pickle \
	--label $LABEL \
	--sampling-seed 42 \
	--outdir $OUTDIR/$LABEL \
	--check-point-deltaT 1800 \
	--nlive 1024 \
	--naccept 60 \