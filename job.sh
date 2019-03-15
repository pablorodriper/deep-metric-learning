#!/usr/bin/env bash
#SBATCH --job-name siamese
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/deep-metric-learning
#SBATCH --output ../logs/%x_%u_%j.out

source /home/grupo06/venv/bin/activate
python src/main.py datasets/tsinghua_resized/ --batch-size 128
