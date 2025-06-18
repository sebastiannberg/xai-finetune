#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=ifi_a2_t2_se0_wb1
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
    --alpha 0.99 \
    --temperature 0.0001 \
    --start_epoch 2 \
    --which_blocks first \
    --sbatch_script ifi_a2_t2_se0_wb1.sh
