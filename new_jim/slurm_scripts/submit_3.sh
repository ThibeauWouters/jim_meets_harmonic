#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output=injection_3.out
#SBATCH --job-name=3

now=$(date)
echo "$now"

# Loading modules
# module load 2024
# module load Python/3.10.4-GCCcore-11.3.0
conda activate /home/twouters2/miniconda3/envs/jim_os

# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

# Run the script
export ID=3
export OUTDIR=./11d_runs/outdir/injection_from_bilby_seed_$ID
python injection_recovery.py \
    --outdir ./11d_runs/outdir/ \
    --waveform-approximant "IMRPhenomD" \
    --id "from_bilby_seed_$ID" \
    --use-scheduler \
    --from-bilby True \
    --n-loop-training 10 \
    --n-loop-production 10 \
    --n-local-steps 10 \
    --n-global-steps 1000 \
    --n-epochs 100 \
    --sampling-seed "$ID" \
    --eps-mass-matrix 0.001 \
    --psd-file-H1 ../psds/psd.txt \
    --psd-file-L1 ../psds/psd.txt \
    --psd-file-V1 ../psds/psd_virgo.txt \

# Move the output file to corresponding directory:
# mv injection_$ID.out $OUTDIR/log.out

echo "DONE"