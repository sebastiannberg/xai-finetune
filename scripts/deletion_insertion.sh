#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=deletion_insertion
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=00:05:00

source venv/bin/activate

python src/deletion_insertion.py
