#!/bin/bash
#SBATCH --mem=128G
#SBATCH --job-name=tr_c1_b00_b11
#SBATCH -t 120:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=logs/c1_b00_b11-%j.out
#SBATCH --error=logs/c1_b00_b11-%j.err

python3 ./train.py 0.75 0.75 1 0 1
