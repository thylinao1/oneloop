#!/bin/bash
#SBATCH --job-name=amex-ladder-emb
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40
#SBATCH --time=2:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=xgpj0
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# Leakage ladder, step 1 of 2 (GPU, short): full-history pooled per-account embeddings from the
# FROZEN backbone. This is the permissive embedding construction that rung L2 removes.
# No pretraining happens here; the checkpoint is read-only.
set -euo pipefail
export HF_HOME=/tmp/$SLURM_JOB_ID/hf XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export TORCH_HOME=/tmp/$SLURM_JOB_ID/torch TRITON_CACHE_DIR=/tmp/$SLURM_JOB_ID/triton
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"

source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
cd "$BASE/scripts/fm"

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch; assert torch.cuda.is_available(), 'FATAL: CUDA unavailable'; print('torch', torch.__version__)"

UEMB=$BASE/eval/user_full_emb.npz
if [ -f "$UEMB" ]; then
  echo "=== $UEMB already present, nothing to do ==="
else
  echo "=== full-history pooled account embeddings $(date -u +%FT%TZ) ==="
  python ladder_eval.py --stage user_emb --prep "$BASE/prep" --ckpt "$BASE/ckpt/final.pt" \
    --out "$UEMB" --batch-size 512 --threads 8 --seed 7
fi
ls -la "$UEMB"
echo "=== EMB DONE $(date -u +%FT%TZ) ==="
