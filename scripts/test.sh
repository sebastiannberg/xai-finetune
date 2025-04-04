#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=test
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:10:00

source venv/bin/activate
python src/evaluate_urban.py
