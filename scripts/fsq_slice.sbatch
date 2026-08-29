#!/bin/bash
#SBATCH --job-name=amex-fsq-slice
#SBATCH --partition=long
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=xgpj0
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out

set -euo pipefail

# STORAGE RULE: all volatile caches to /tmp/$SLURM_JOB_ID; only results/logs to $HOME.
SCRATCH=/tmp/$SLURM_JOB_ID
mkdir -p "$SCRATCH"/hf "$SCRATCH"/xdg "$SCRATCH"/torch "$SCRATCH"/triton "$SCRATCH"/duckdb_ext "$SCRATCH"/duckdb_tmp
export HF_HOME=$SCRATCH/hf
export XDG_CACHE_HOME=$SCRATCH/xdg
export TORCH_HOME=$SCRATCH/torch
export TRITON_CACHE_DIR=$SCRATCH/triton
export TMPDIR=$SCRATCH
export PIP_CACHE_DIR=$SCRATCH/xdg/pip

trap 'rm -rf "$SCRATCH"' EXIT

echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"
python3 -m venv "$SCRATCH/venv"
"$SCRATCH/venv/bin/pip" -q install --no-cache-dir duckdb
"$SCRATCH/venv/bin/python" -c "import duckdb; print('[sbatch] duckdb', duckdb.__version__)"
"$SCRATCH/venv/bin/python" "$HOME/amex-oneloop/fsq_slice.py"
echo "[sbatch] exit OK"
