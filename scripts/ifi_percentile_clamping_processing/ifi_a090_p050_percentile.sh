#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=ifi_a090_p050_percentile
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source venv/bin/activate

python src/main.py \
    --mode ifi \
    --lr 1e-05 \
    --weight_decay 1e-05 \
    --batch_size 8 \
    --alpha 0.90 \
    --grad_processing_mode percentile_clamping \
    --percentile 0.5 \
    --sbatch_script ifi_a090_p050_percentile.sh
