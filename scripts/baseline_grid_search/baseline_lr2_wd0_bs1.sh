#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=baseline_lr2_wd0_bs1
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source venv/bin/activate

python src/main.py \
    --mode baseline \
    --lr 1e-6 \
    --weight_decay 1e-3 \
    --batch_size 16 \
    --save_frequency 30 \
    --sbatch_script baseline_lr2_wd0_bs1.sh
