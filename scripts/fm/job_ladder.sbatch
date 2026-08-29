#!/bin/bash
#SBATCH --job-name=amex-ladder
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --exclude=xgpj0
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# The leakage ladder. EVALUATION ONLY: one frozen checkpoint, no pretraining.
# 1) GPU pass: full-history pooled per-account embeddings (the permissive construction)
# 2) code-path smoke (tiny, fail-fast)
# 3) the four rungs -> ladder.json (asserts L3 reproduces the shipped numbers)
# 4) --check: recompute and compare every numeric leaf at 1e-6
set -euo pipefail

export HF_HOME=/tmp/$SLURM_JOB_ID/hf
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export TORCH_HOME=/tmp/$SLURM_JOB_ID/torch
export TRITON_CACHE_DIR=/tmp/$SLURM_JOB_ID/triton
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"

source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
FM=$BASE/scripts/fm
WORK=/tmp/$SLURM_JOB_ID/work
mkdir -p "$WORK"
cd "$FM"

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch; assert torch.cuda.is_available(), 'FATAL: CUDA unavailable'; print('torch', torch.__version__)"

UEMB=$BASE/eval/user_full_emb.npz
if [ ! -f "$UEMB" ]; then
  echo "=== STEP 1: full-history pooled account embeddings (GPU) $(date -u +%FT%TZ) ==="
  python ladder_eval.py --stage user_emb --prep "$BASE/prep" --ckpt "$BASE/ckpt/final.pt" \
    --out "$UEMB" --batch-size 512 --threads 16 --seed 7
else
  echo "=== STEP 1: $UEMB already present, reusing ==="
fi

COMMON=(--stage run --prep "$BASE/prep" --eval "$BASE/eval" --asof "$BASE/eval/asof.npy"
        --user-emb "$UEMB" --merchants "$BASE/merchant_embeddings.parquet" --threads 16 --seed 7)

echo "=== STEP 2: code-path smoke $(date -u +%FT%TZ) ==="
python ladder_eval.py "${COMMON[@]}" --out "$WORK/smoke_ladder.json" \
  --smoke 40000 --bootstrap 20 --fraud-estimators 40 --mcc-estimators 20
python -c "
import json,sys
d=json.load(open('$WORK/smoke_ladder.json'))
assert [r['rung'] for r in d['rungs']]==['L0','L1','L2','L3'], d.keys()
assert d['l0_shared_accounts']>0
print('[smoke] four rungs present, L0 shares', d['l0_shared_accounts'], 'accounts')"
echo "=== SMOKE PASSED $(date -u +%FT%TZ) ==="

echo "=== STEP 3: the four rungs, full size $(date -u +%FT%TZ) ==="
# exit 4 = L3 did not reproduce the shipped numbers. The file is still written, as
# obtained, and step 4 still runs; the log carries the measured difference either way.
set +e
python ladder_eval.py "${COMMON[@]}" --out "$BASE/ladder.json" \
  --shipped "$BASE/shipped_backbone_transfer.json" --bootstrap 1000
RC3=$?
set -e
echo "=== STEP 3 DONE (exit $RC3) $(date -u +%FT%TZ) ==="
test "$RC3" -eq 0 -o "$RC3" -eq 4

echo "=== STEP 4: --check reproduction at 1e-6 $(date -u +%FT%TZ) ==="
set +e
python ladder_eval.py "${COMMON[@]}" --out "$BASE/ladder.json" \
  --shipped "$BASE/shipped_backbone_transfer.json" --bootstrap 1000 --check
RC4=$?
set -e
echo "=== STEP 4 DONE (exit $RC4) $(date -u +%FT%TZ) ==="

echo "=== ALL DONE (step3=$RC3 step4=$RC4) $(date -u +%FT%TZ) ==="
ls -la "$BASE/ladder.json"
