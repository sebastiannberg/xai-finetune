#!/bin/bash
#SBATCH --job-name=finetune_urban
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=05:00:00

source venv/bin/activate
python finetune_urban.py
