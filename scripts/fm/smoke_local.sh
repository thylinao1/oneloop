#!/bin/bash
# Local Mac smoke: <=50k rows, 1 thread, tiny model, CPU. END-TO-END -> valid JSON.
# Usage: scripts/fm/smoke_local.sh <scratch_dir>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRATCH="${1:?usage: smoke_local.sh <scratch_dir>}"
PY="$ROOT/.venv/bin/python"
FM="$ROOT/scripts/fm"
mkdir -p "$SCRATCH"
export OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SAMPLE="$SCRATCH/sample.csv"
if [ ! -f "$SAMPLE" ]; then
  # first 2M raw rows (~160 users), prep then strides to 50k
  tar -xzOf "$ROOT/data/transactions.tgz" 2>/dev/null | head -2000001 > "$SAMPLE" || true
  wc -l "$SAMPLE"
fi

rm -rf "$SCRATCH/ckpt" "$SCRATCH/eval"   # smoke always starts clean
cd "$FM"
"$PY" prep.py --csv "$SAMPLE" --out "$SCRATCH/prep" --max-rows 2000000 --sample-stride 40 --seed 7
"$PY" pretrain.py --prep "$SCRATCH/prep" --ckpt "$SCRATCH/ckpt" \
  --d-model 32 --layers 2 --heads 2 --ff 64 --window 10 --stride 5 \
  --epochs 1 --max-steps 120 --batch-size 64 --workers 0 --threads 1 --device cpu \
  --warmup 15 --lr 1e-3 --seed 7 --assert-improve
"$PY" transfer_eval.py --stage select --prep "$SCRATCH/prep" --out "$SCRATCH/eval" \
  --max-train 20000 --max-test 10000 --seed 7
"$PY" embed.py --prep "$SCRATCH/prep" --ckpt "$SCRATCH/ckpt/final.pt" \
  --merchants "$SCRATCH/eval/merchant_embeddings.parquet" \
  --asof "$SCRATCH/eval/eval_rows.npy" --asof-out "$SCRATCH/eval/asof.npy" \
  --device cpu --threads 1 --batch-size 256 --seed 7
"$PY" transfer_eval.py --stage run --prep "$SCRATCH/prep" --eval "$SCRATCH/eval" \
  --asof "$SCRATCH/eval/asof.npy" --merchants "$SCRATCH/eval/merchant_embeddings.parquet" \
  --ckpt-summary "$SCRATCH/ckpt/pretrain_summary.json" \
  --out "$SCRATCH/backbone_transfer.json" --bootstrap 30 --mcc-estimators 15 --threads 1 --seed 7
"$PY" validate_json.py "$SCRATCH/backbone_transfer.json"
echo "LOCAL SMOKE PASSED"
