#!/usr/bin/env bash
#SBATCH --job-name deep-metric-learning
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --qos masterhigh
#SBATCH --partition mhigh
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/deep-metric-learning
#SBATCH --output logs/%x_%j.out

source /home/grupo06/venv/bin/activate
python src/main.py --dataset_dir datasets/tsinghua_resized --batch_size 128