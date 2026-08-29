#!/bin/bash
#SBATCH --job-name=amex-merchant
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
# Merchant-axis leg (partial two-sided demonstration): the SAME tokenized events
# regrouped per merchant, time-ordered (thin wrapper regroup_merchant.py; fm
# tokenization reused untouched). Produces merchant-axis backbone + transfer
# eval (entity = merchant) + $BASE/merchant_embeddings_v2.parquet.
export TAG=merchant-axis MODE=merchant ROWS=0 SEED_MODEL=7 SEED_SELECT=7 AXIS=merchant
export NOTE="partial two-sided demonstration: same events regrouped per merchant (time-ordered) via thin wrapper; sequence entity = merchant throughout pretrain + transfer eval; disclosed caveat: cardholders cross merchant boundaries, split is merchant-disjoint only; config/seed identical to the main 24M run (seed 7)"
exec bash "$HOME/amex-oneloop/scripts/scale/scale_pipeline.sh"
