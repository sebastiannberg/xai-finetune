#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=ifi_a90_t10-5_se2_wball
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source venv/bin/activate

python src/main.py \
    --mode ifi \
    --lr 1e-5 \
    --weight_decay 1e-5 \
    --batch_size 8 \
    --alpha 0.9 \
    --temperature 1e-05 \
    --start_epoch 2 \
    --which_blocks all \
    --save_frequency 60 \
    --sbatch_script ifi_a90_t10-5_se2_wball.sh
