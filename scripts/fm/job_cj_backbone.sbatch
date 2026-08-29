#!/bin/bash
#SBATCH --job-name=amex-cj-backbone
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=xgpj0
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# Complete Journey backbone: in-job SMOKE (fail-fast) -> full prep -> pretrain seeds 7+8
# -> select -> embeddings (stores + as-of + pre-cut household pooling). The corpus is
# 2.6M rows (vs TabFormer's 24M), so both 8-epoch pretrains together are well under an
# hour of GPU; the 6h limit is headroom, not an estimate. Resubmit-safe: pretrain resumes
# from latest.pt, prep skips if done.
# Storage rules: volatile caches to /tmp/$SLURM_JOB_ID; prep/ckpt/embeddings to $HOME.
# The UNMODIFIED pretrain.py/embed.py run through cj_run.py (field-list patch only).
set -euo pipefail

# duplicate-run guard: whichever amex-cj-backbone starts second exits at once
# (checkpoint/prep dirs are shared)
OTHER=$(squeue --me -h -t RUNNING -o "%i %j" | awk -v me="$SLURM_JOB_ID" '$2 ~ /^amex-cj-backbone/ && $1 != me {print $1; exit}')
if [ -n "${OTHER:-}" ]; then
  echo "duplicate cj backbone job $OTHER already RUNNING - exiting cleanly"
  exit 0
fi

export HF_HOME=/tmp/$SLURM_JOB_ID/hf
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export TORCH_HOME=/tmp/$SLURM_JOB_ID/torch
export TRITON_CACHE_DIR=/tmp/$SLURM_JOB_ID/triton
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"

source "$HOME/amex-oneloop/venv/bin/activate"   # venv built on a compute node (login-node 1GB ulimit)
BASE=$HOME/amex-oneloop
FM=$BASE/scripts/fm
CJ=$BASE/cj
WORK=/tmp/$SLURM_JOB_ID/work
mkdir -p "$WORK" "$CJ/ckpt" "$CJ/eval" "$BASE/logs"
cd "$FM"

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
python -c "import torch, lightgbm, polars, pyarrow, numpy, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available(), 'FATAL: CUDA unavailable - fail fast'"

# provenance gate: no training before the download job verified the corpus
for f in "$CJ/transaction_data.parquet" "$CJ/product.parquet" "$CJ/meta.json"; do
  test -f "$f" || { echo "FATAL: $f missing - run job_cj_download.sbatch first"; exit 2; }
done

# ---------------- IN-JOB SMOKE (minutes, hard asserts, fail-fast) ----------------
SMOKE=$WORK/smoke
rm -rf "$SMOKE"; mkdir -p "$SMOKE"
echo "=== SMOKE: prep_cj ==="
python prep_cj.py --txn "$CJ/transaction_data.parquet" --product "$CJ/product.parquet" \
  --out "$SMOKE/prep" --max-rows 400000 --sample-stride 4 --seed 7 --skip-sha
echo "=== SMOKE: pretrain (tiny, assert loss decreases + finite) ==="
python cj_run.py pretrain --prep "$SMOKE/prep" --ckpt "$SMOKE/ckpt" \
  --d-model 64 --layers 2 --heads 4 --ff 128 --window 10 --stride 5 \
  --epochs 1 --max-steps 300 --batch-size 128 --workers 2 --warmup 30 --lr 1e-3 \
  --seed 7 --assert-improve
echo "=== SMOKE: select + embed + transfer ==="
python transfer_eval_cj.py --stage select --prep "$SMOKE/prep" --out "$SMOKE/eval" \
  --max-train 30000 --max-test 15000 --seed 7
python cj_run.py embed --prep "$SMOKE/prep" --ckpt "$SMOKE/ckpt/final.pt" \
  --merchants "$SMOKE/eval/store_embeddings.parquet" \
  --asof "$SMOKE/eval/eval_rows.npy" --asof-out "$SMOKE/eval/asof.npy" \
  --batch-size 512 --seed 7
python transfer_eval_cj.py --stage run --prep "$SMOKE/prep" --eval "$SMOKE/eval" \
  --asof "$SMOKE/eval/asof.npy" --ckpt-summary "$SMOKE/ckpt/pretrain_summary.json" \
  --out "$SMOKE/cj_transfer_smoke.json" --bootstrap 50 --mcc-estimators 30 --seed 42
python -c "
import json
d=json.load(open('$SMOKE/cj_transfer_smoke.json'))
t=d['tasks']['next_commodity']
for k in ('baseline_top1','with_emb_top1','delta_top1_ci','majority_floor_top1'):
    assert k in t, f'smoke json missing {k}'
assert d['labels']==['real']
print('[smoke] transfer json shape OK')"
echo "=== SMOKE PASSED $(date -u +%FT%TZ) ==="

# ---------------- FULL PIPELINE ----------------
echo "=== FULL: prep_cj (2.6M rows; skipped if already done) ==="
python prep_cj.py --txn "$CJ/transaction_data.parquet" --product "$CJ/product.parquet" \
  --out "$CJ/prep" --seed 7 --skip-if-done

for SEED in 7 8; do
  echo "=== FULL: pretrain seed $SEED (resume-aware) ==="
  python cj_run.py pretrain --prep "$CJ/prep" --ckpt "$CJ/ckpt/s$SEED" \
    --d-model 512 --layers 6 --heads 8 --ff 2048 --window 16 --stride 8 \
    --epochs 8 --batch-size 256 --workers 8 --seed "$SEED" --ckpt-minutes 30 --resume auto
done

echo "=== FULL: select eval rows (one pre-registered split for both seeds) ==="
python transfer_eval_cj.py --stage select --prep "$CJ/prep" --out "$CJ/eval" --seed 7

for SEED in 7 8; do
  echo "=== FULL: embeddings for seed $SEED (stores + as-of + household pre-cut) ==="
  python cj_run.py embed --prep "$CJ/prep" --ckpt "$CJ/ckpt/s$SEED/final.pt" \
    --merchants "$CJ/store_embeddings_s$SEED.parquet" \
    --asof "$CJ/eval/eval_rows.npy" --asof-out "$CJ/eval/asof_s$SEED.npy" \
    --batch-size 512 --seed "$SEED"
  python cj_offer_head.py --stage emb --prep "$CJ/prep" \
    --ckpt "$CJ/ckpt/s$SEED/final.pt" --out "$CJ/hh_emb_s$SEED.npz" \
    --device cuda --batch-size 512 --seed "$SEED"
done

echo "=== ALL DONE $(date -u +%FT%TZ) ==="
ls -la "$CJ" "$CJ/eval" "$CJ/ckpt"/s7 "$CJ/ckpt"/s8
