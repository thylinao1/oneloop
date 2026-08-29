#!/bin/bash
#SBATCH --job-name=amex-ladder
#SBATCH --partition=long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# The leakage ladder. EVALUATION ONLY: one frozen checkpoint, no pretraining,
# and no GPU (the gpu-long queue estimated a 22h wait; the forward pass is affordable on
# CPU and the downstream fits are CPU work anyway).
#   1) full-history pooled per-account embeddings (the permissive construction L2 removes)
#   2) code-path smoke (tiny, fail-fast)
#   3) the four rungs -> ladder.json, asserting L0 really does share accounts across the
#      split and that L3 reproduces the shipped transfer numbers
#   4) --check: recompute and compare every numeric leaf at 1e-6
set -euo pipefail
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
mkdir -p "$XDG_CACHE_HOME"

source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
WORK=/tmp/$SLURM_JOB_ID/work
mkdir -p "$WORK"
cd "$BASE/scripts/fm"

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
free -g | head -2 || true
nproc

UEMB=$BASE/eval/user_full_emb.npz
if [ -f "$UEMB" ]; then
  echo "=== STEP 1: $UEMB already present, reusing ==="
else
  echo "=== STEP 1: full-history pooled account embeddings (CPU) $(date -u +%FT%TZ) ==="
  python ladder_eval.py --stage user_emb --prep "$BASE/prep" --ckpt "$BASE/ckpt/final.pt" \
    --out "$WORK/user_full_emb.npz" --device cpu --batch-size 256 --threads 24 --seed 7
  mv "$WORK/user_full_emb.npz" "$UEMB"
fi
ls -la "$UEMB"
echo "=== STEP 1 DONE $(date -u +%FT%TZ) ==="

COMMON=(--stage run --prep "$BASE/prep" --eval "$BASE/eval" --asof "$BASE/eval/asof.npy"
        --user-emb "$UEMB" --merchants "$BASE/merchant_embeddings.parquet"
        --threads 24 --seed 7)

echo "=== STEP 2: code-path smoke $(date -u +%FT%TZ) ==="
python ladder_eval.py "${COMMON[@]}" --out "$WORK/smoke_ladder.json" \
  --smoke 40000 --bootstrap 20 --fraud-estimators 40 --mcc-estimators 20
python -c "
import json
d=json.load(open('$WORK/smoke_ladder.json'))
assert [r['rung'] for r in d['rungs']]==['L0','L1','L2','L3']
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
