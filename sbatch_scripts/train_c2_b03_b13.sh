#!/bin/bash
#SBATCH --mem=128G
#SBATCH --job-name=tr_c2_b03_b13
#SBATCH -t 120:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=logs/c2_b03_b13-%j.out
#SBATCH --error=logs/c2_b03_b13-%j.err

python3 ./train.py 0.75 0.75 2 3 3
