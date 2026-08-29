#!/bin/bash
# scale_pipeline.sh; shared body for the scaling-curve pretrain jobs.
# Invoked by job_*.sbatch with env: TAG, MODE(subset|full24m|merchant), ROWS,
# SEED_MODEL, SEED_SELECT, AXIS, NOTE.
# Reuses scripts/fm/ untouched (prep/pretrain/embed/transfer_eval/validate_json);
# thin wrappers live in scripts/scale/. Storage: volatile work + caches in
# /tmp/$SLURM_JOB_ID; only ckpt + transfer.json (+ merchant v2 parquet) to $HOME.
set -euo pipefail

export HF_HOME=/tmp/$SLURM_JOB_ID/hf
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export TORCH_HOME=/tmp/$SLURM_JOB_ID/torch
export TRITON_CACHE_DIR=/tmp/$SLURM_JOB_ID/triton
export VLLM_CACHE_ROOT=/tmp/$SLURM_JOB_ID/vllm
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"

source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
FM=$BASE/scripts/fm
SC=$BASE/scripts/scale
OUT=$BASE/scale/$TAG
WORK=/tmp/$SLURM_JOB_ID/work
mkdir -p "$OUT/ckpt" "$WORK" "$BASE/logs"
cd "$FM"

echo "=== $TAG | mode=$MODE | node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
python -c "import torch, lightgbm, polars, numpy, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available(), 'FATAL: CUDA unavailable; fail fast'"

# ---------------- data / prep per mode ----------------
case "$MODE" in
  subset)
    echo "=== extracting TabFormer CSV ==="
    tar -xzf "$BASE/data/transactions.tgz" -C "$WORK"
    echo "=== earliest-time subset: $ROWS rows ==="
    python -u "$SC/make_subset.py" --csv "$WORK/card_transaction.v1.csv" --rows "$ROWS" --out "$WORK/subset.csv"
    echo "=== prep (unmodified fm/prep.py on the subset) ==="
    python -u prep.py --csv "$WORK/subset.csv" --out "$WORK/prep" --seed 7
    PREP=$WORK/prep
    ;;
  full24m)
    # identical deterministic prep already on disk from the main backbone job; reuse read-only
    echo "=== reusing \$BASE/prep (full 24M, deterministic, read-only) ==="
    test -f "$BASE/prep/meta.json"
    PREP=$BASE/prep
    ;;
  merchant)
    echo "=== regrouping \$BASE/prep by merchant (thin wrapper; fm code untouched) ==="
    test -f "$BASE/prep/meta.json"
    python -u "$SC/regroup_merchant.py" --prep-in "$BASE/prep" --out "$WORK/prep"
    PREP=$WORK/prep
    ;;
  *) echo "unknown MODE=$MODE"; exit 2 ;;
esac

# ---------------- in-job fail-fast smoke (~3 min) ----------------
echo "=== SMOKE: tiny pretrain, assert loss decreases + finite ==="
python -u pretrain.py --prep "$PREP" --ckpt "$WORK/smoke_ckpt" \
  --d-model 64 --layers 2 --heads 4 --ff 128 --window 10 --stride 5 \
  --epochs 1 --max-steps 250 --batch-size 128 --workers 2 --warmup 30 --lr 1e-3 \
  --seed 7 --assert-improve --resume never
echo "=== SMOKE PASSED $(date -u +%FT%TZ) ==="

# ---------------- full pretrain (same config as main 24M run; epoch ckpts) ----------------
echo "=== FULL pretrain (seed $SEED_MODEL) ==="
python -u pretrain.py --prep "$PREP" --ckpt "$OUT/ckpt" \
  --d-model 512 --layers 6 --heads 8 --ff 2048 --window 16 --stride 8 \
  --epochs 8 --batch-size 256 --workers 8 --seed "$SEED_MODEL" --ckpt-minutes 30 --resume auto

# ---------------- transfer eval (same task shapes as backbone_transfer.json) ----------------
echo "=== select (seed $SEED_SELECT) ==="
python -u transfer_eval.py --stage select --prep "$PREP" --out "$WORK/eval" \
  --max-train 800000 --max-test 300000 --seed "$SEED_SELECT"

echo "=== embeddings (merchant pooled + as-of) ==="
python -u embed.py --prep "$PREP" --ckpt "$OUT/ckpt/final.pt" \
  --merchants "$WORK/eval/merchant_embeddings.parquet" \
  --asof "$WORK/eval/eval_rows.npy" --asof-out "$WORK/eval/asof.npy" \
  --batch-size 512 --seed "$SEED_MODEL"

if [ "$MODE" = "merchant" ]; then
  cp "$WORK/eval/merchant_embeddings.parquet" "$BASE/merchant_embeddings_v2.parquet"
  echo "=== merchant-axis embeddings persisted -> \$BASE/merchant_embeddings_v2.parquet ==="
fi

echo "=== transfer eval run ==="
python -u transfer_eval.py --stage run --prep "$PREP" --eval "$WORK/eval" \
  --asof "$WORK/eval/asof.npy" --merchants "$WORK/eval/merchant_embeddings.parquet" \
  --ckpt-summary "$OUT/ckpt/pretrain_summary.json" \
  --out "$OUT/backbone_transfer_raw.json" --bootstrap 1000 --threads 16 --seed "$SEED_MODEL"
python -u validate_json.py "$OUT/backbone_transfer_raw.json"

echo "=== scaling entry ==="
python -u "$SC/make_scaling_entry.py" --transfer "$OUT/backbone_transfer_raw.json" \
  --prep-meta "$PREP/meta.json" --axis "$AXIS" --tag "$TAG" --note "$NOTE" \
  --out "$OUT/transfer.json"

echo "=== $TAG ALL DONE $(date -u +%FT%TZ) ==="
ls -la "$OUT"
