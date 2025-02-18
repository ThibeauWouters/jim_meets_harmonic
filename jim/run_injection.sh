#!/bin/bash

for i in {0..9}
do
  python injection_recovery.py \
    --outdir "./11d_runs/outdir/" \
    --id "from_bilby_seed_$i" \
    --from-bilby True \
    --psd-file-H1 "../psds/psd.txt" \
    --psd-file-L1 "../psds/psd.txt" \
    --psd-file-V1 "../psds/psd_virgo.txt" \
    --use-relative-binning False \
    --n-loop-training 10 \
    --n-loop-production 5 \
    --n-global-steps 1000 \
    --n-local-steps 10 \
    --eps-mass-matrix 0.001 \
    --n-epochs 100 \
    --sampling-seed "$i"
done
