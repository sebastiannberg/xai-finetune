#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=et_with_augmentations_se10_k0
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source venv/bin/activate

python src/main.py \
    --mode et \
    --lr 1e-05 \
    --weight_decay 1e-05 \
    --batch_size 8 \
    --start_epoch 10 \
    --sigma_k 0.0 \
    --mixup_prob 0.5 \
    --roll_mag_aug \
    --freqm 48 \
    --timem 192 \
    --save_frequency 60 \
    --sbatch_script et_with_augmentations_se10_k0.sh
