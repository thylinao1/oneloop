#!/bin/bash
#SBATCH --job-name=amex-scale3m
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-47
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=xgpj0
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# Scaling-curve point: 3M earliest-time rows, cardholder axis, main-run config.
# (h100-47 binds one visible device; fine, this is a single-GPU job.)
export TAG=scale-3m MODE=subset ROWS=3000000 SEED_MODEL=7 SEED_SELECT=7 AXIS=cardholder
export NOTE="scaling point: earliest-time-consistent 3M-row subset; time cut (0.82 quantile), vocab and amount edges recomputed pre-cut WITHIN the subset via unmodified fm/prep.py; params/config/seed identical to the main 24M run (seed 7)"
exec bash "$HOME/amex-oneloop/scripts/scale/scale_pipeline.sh"
