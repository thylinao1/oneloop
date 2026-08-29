#!/bin/bash
#SBATCH --job-name=amex-narratives
#SBATCH --partition=gpu
#SBATCH --gpus=h200-141
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --exclude=xgpj0
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=%x-%j.out

set -euo pipefail

# ALL caches to node-local scratch (hidden ~500GB home quota)
SCRATCH=/tmp/$SLURM_JOB_ID
mkdir -p "$SCRATCH"
export HF_HOME="$SCRATCH/hf"
export XDG_CACHE_HOME="$SCRATCH/xdg"
export TORCH_HOME="$SCRATCH/torch"
export TRITON_CACHE_DIR="$SCRATCH/triton"
export VLLM_CACHE_ROOT="$SCRATCH/vllm"
export PYTHONUNBUFFERED=1

source ~/miniconda3/bin/activate csa-vllm

WORK=$HOME/amex-oneloop/narratives
cd "$WORK"

python -u generate.py \
  --inputs narratives_inputs.json \
  --out narratives.json \
  --model Qwen/Qwen3-32B

echo "[job] done; narratives.json:"
ls -la narratives.json
