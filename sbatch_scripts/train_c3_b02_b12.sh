#!/bin/bash
#SBATCH --mem=128G
#SBATCH --job-name=tr_c3_b02_b12
#SBATCH -t 120:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=logs/c3_b02_b12-%j.out
#SBATCH --error=logs/c3_b02_b12-%j.err

python3 ./train.py 0.75 0.75 3 2 2
