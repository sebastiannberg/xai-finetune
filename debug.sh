#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=debug
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00

source venv/bin/activate
python src/debug.py
