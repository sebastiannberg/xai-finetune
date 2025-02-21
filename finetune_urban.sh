#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=finetune_urban
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=04:00:00

source venv/bin/activate
python finetune_urban.py
