#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=et_se1_k0p5
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
    --start_epoch 1 \
    --sigma_k 0.5 \
    --sbatch_script et_se1_k0p5.sh
