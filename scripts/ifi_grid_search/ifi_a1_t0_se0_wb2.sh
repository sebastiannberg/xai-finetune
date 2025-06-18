#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=ifi_a1_t0_se0_wb2
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
    --weight_decay 0.0005 \
    --batch_size 8 \
    --alpha 0.95 \
    --temperature 1e-06 \
    --start_epoch 2 \
    --which_blocks last \
    --sbatch_script ifi_a1_t0_se0_wb2.sh
