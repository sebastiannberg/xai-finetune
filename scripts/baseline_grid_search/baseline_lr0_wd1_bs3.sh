#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=baseline_lr0_wd1_bs3
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source venv/bin/activate

python src/main.py \
    --mode baseline \
    --lr 1e-4 \
    --weight_decay 5e-4 \
    --batch_size 64 \
    --sbatch_script baseline_lr0_wd1_bs3.sh
