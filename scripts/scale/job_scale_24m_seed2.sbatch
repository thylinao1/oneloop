#!/bin/bash
#SBATCH --job-name=amex-scale24m-s2
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-96
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --exclude=xgpj0
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# Seed-stability point: full 24M rows, cardholder axis, seed 1337 for model init
# + eval fit; eval ROW SELECTION pinned to the main run's seed 7 so the test set
# is identical and only the backbone seed varies. Reuses $BASE/prep read-only
# (prep is deterministic; no RNG touches its outputs).
export TAG=scale-24m-seed2 MODE=full24m ROWS=0 SEED_MODEL=1337 SEED_SELECT=7 AXIS=cardholder
export NOTE="seed-stability point: full 24M rows; model init/embed/eval-fit/bootstrap seed 1337; eval row selection pinned to main-run seed 7 (identical test rows); config identical to the main 24M run"
exec bash "$HOME/amex-oneloop/scripts/scale/scale_pipeline.sh"
