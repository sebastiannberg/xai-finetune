#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=baseline_without_augmentations_lr10-5_wd10-5_bs8
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source venv/bin/activate

python src/main.py \
    --mode baseline \
    --lr 1e-5 \
    --weight_decay 1e-5 \
    --batch_size 8 \
    --save_frequency 60 \
    --sbatch_script baseline_without_augmentations_lr10-5_wd10-5_bs8.sh
