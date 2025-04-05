#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=baseline
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --time=01:00:00

cd "$(dirname "$0")/.."
source venv/bin/activate
python src/main.py --mode baseline
