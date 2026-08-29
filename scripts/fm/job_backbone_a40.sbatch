#!/bin/bash
#SBATCH --job-name=amex-backbone40
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --exclude=xgpj0
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# ONE chained job: in-job SMOKE (fail-fast) -> full prep -> pretrain -> embed -> transfer_eval.
# Storage rules: all volatile caches to /tmp/$SLURM_JOB_ID; only prep/ckpt/results/logs to $HOME/amex-oneloop.
set -euo pipefail

# duplicate-run guard: a100-80 + a100-40 variants of this job may both be queued;
# whichever starts second exits at once (checkpoint/prep dirs are shared).
OTHER=$(squeue --me -h -t RUNNING -o "%i %j" | awk -v me="$SLURM_JOB_ID" '$2 ~ /^amex-backbone/ && $1 != me {print $1; exit}')
if [ -n "${OTHER:-}" ]; then
  echo "duplicate backbone job $OTHER already RUNNING; exiting cleanly"
  exit 0
fi

export HF_HOME=/tmp/$SLURM_JOB_ID/hf
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export TORCH_HOME=/tmp/$SLURM_JOB_ID/torch
export TRITON_CACHE_DIR=/tmp/$SLURM_JOB_ID/triton
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"

source "$HOME/amex-oneloop/venv/bin/activate"   # venv built on a compute node (login-node 1GB ulimit kills conda solves)
BASE=$HOME/amex-oneloop
FM=$BASE/scripts/fm
WORK=/tmp/$SLURM_JOB_ID/work
mkdir -p "$WORK" "$BASE/ckpt" "$BASE/eval" "$BASE/logs"
cd "$FM"

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
python -c "import torch, lightgbm, polars, pyarrow, numpy, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available(), 'FATAL: CUDA unavailable (driver vs cu130 wheel?); fail fast'"

echo "=== extracting TabFormer CSV to node-local /tmp ==="
tar -xzf "$BASE/data/transactions.tgz" -C "$WORK"
CSV=$WORK/card_transaction.v1.csv
wc -l "$CSV"

# ---------------- IN-JOB SMOKE (~5 min, hard asserts, fail-fast) ----------------
SMOKE=$WORK/smoke
rm -rf "$SMOKE"; mkdir -p "$SMOKE"
echo "=== SMOKE: prep ==="
python prep.py --csv "$CSV" --out "$SMOKE/prep" --max-rows 2000000 --sample-stride 10 --seed 7
echo "=== SMOKE: pretrain (tiny, assert loss decreases + finite) ==="
python pretrain.py --prep "$SMOKE/prep" --ckpt "$SMOKE/ckpt" \
  --d-model 64 --layers 2 --heads 4 --ff 128 --window 10 --stride 5 \
  --epochs 1 --max-steps 300 --batch-size 128 --workers 2 --warmup 30 --lr 1e-3 --seed 7 --assert-improve
echo "=== SMOKE: select + embed + transfer ==="
python transfer_eval.py --stage select --prep "$SMOKE/prep" --out "$SMOKE/eval" \
  --max-train 30000 --max-test 15000 --seed 7
python embed.py --prep "$SMOKE/prep" --ckpt "$SMOKE/ckpt/final.pt" \
  --merchants "$SMOKE/eval/merchant_embeddings.parquet" \
  --asof "$SMOKE/eval/eval_rows.npy" --asof-out "$SMOKE/eval/asof.npy" --batch-size 512 --seed 7
python transfer_eval.py --stage run --prep "$SMOKE/prep" --eval "$SMOKE/eval" \
  --asof "$SMOKE/eval/asof.npy" --merchants "$SMOKE/eval/merchant_embeddings.parquet" \
  --ckpt-summary "$SMOKE/ckpt/pretrain_summary.json" \
  --out "$SMOKE/backbone_transfer.json" --bootstrap 50 --mcc-estimators 30 --seed 7
python validate_json.py "$SMOKE/backbone_transfer.json"
echo "=== SMOKE PASSED $(date -u +%FT%TZ) ==="

# ---------------- FULL PIPELINE ----------------
echo "=== FULL: prep (24M rows; skipped if already done) ==="
python prep.py --csv "$CSV" --out "$BASE/prep" --seed 7 --skip-if-done

echo "=== FULL: pretrain (resume-aware) ==="
python pretrain.py --prep "$BASE/prep" --ckpt "$BASE/ckpt" \
  --d-model 512 --layers 6 --heads 8 --ff 2048 --window 16 --stride 8 \
  --epochs 8 --batch-size 256 --workers 8 --seed 7 --ckpt-minutes 30 --resume auto

echo "=== FULL: select eval rows ==="
python transfer_eval.py --stage select --prep "$BASE/prep" --out "$BASE/eval" \
  --max-train 800000 --max-test 300000 --seed 7

echo "=== FULL: embeddings (merchant + as-of) ==="
python embed.py --prep "$BASE/prep" --ckpt "$BASE/ckpt/final.pt" \
  --merchants "$BASE/merchant_embeddings.parquet" \
  --asof "$BASE/eval/eval_rows.npy" --asof-out "$BASE/eval/asof.npy" --batch-size 512 --seed 7

echo "=== FULL: transfer eval ==="
python transfer_eval.py --stage run --prep "$BASE/prep" --eval "$BASE/eval" \
  --asof "$BASE/eval/asof.npy" --merchants "$BASE/merchant_embeddings.parquet" \
  --ckpt-summary "$BASE/ckpt/pretrain_summary.json" \
  --out "$BASE/backbone_transfer.json" --bootstrap 1000 --threads 16 --seed 7
python validate_json.py "$BASE/backbone_transfer.json"

echo "=== ALL DONE $(date -u +%FT%TZ) ==="
ls -la "$BASE"
