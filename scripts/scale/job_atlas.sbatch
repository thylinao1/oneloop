#!/bin/bash
#SBATCH --job-name=amex-atlas
#SBATCH --partition=long
#SBATCH --time=42:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=xgpj0
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# Atlas chain (CPU): waits for the main backbone job's outputs, then projects
# $BASE/merchant_embeddings.parquet to 2D -> $BASE/atlas.json (CONTRACT §2).
# Submit with: sbatch --dependency=afterany:<backbone-jobid> job_atlas.sbatch
# afterany + an in-job gate (below) covers the guarded-hedge takeover case:
# if the primary fails and the a100-80 hedge reruns the pipeline, this job
# simply waits for backbone_transfer.json (written AFTER the merchant parquet,
# atomically) for up to 30h before giving up.
set -euo pipefail

export HF_HOME=/tmp/$SLURM_JOB_ID/hf
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export TORCH_HOME=/tmp/$SLURM_JOB_ID/torch
export TRITON_CACHE_DIR=/tmp/$SLURM_JOB_ID/triton
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"

source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
FM=$BASE/scripts/fm
SC=$BASE/scripts/scale
WORK=/tmp/$SLURM_JOB_ID/work
mkdir -p "$WORK" "$BASE/logs" "$BASE/scale/atlas"

echo "=== atlas | node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="

GATE=$BASE/backbone_transfer.json
for i in $(seq 1 180); do
  if [ -f "$GATE" ]; then break; fi
  echo "[gate] waiting for $GATE ($i/180, 10-min intervals) $(date -u +%FT%TZ)"
  sleep 600
done
if [ ! -f "$GATE" ]; then
  echo "[gate] backbone_transfer.json never appeared after 30h; backbone pipeline did not finish"
  exit 1
fi
test -f "$BASE/merchant_embeddings.parquet"
echo "[gate] backbone outputs present $(date -u +%FT%TZ)"

echo "=== extracting TabFormer CSV (for modal MCC per merchant) ==="
tar -xzf "$BASE/data/transactions.tgz" -C "$WORK"

echo "=== attempting UMAP install into /tmp (PCA fallback handled in-script) ==="
pip install -q --no-cache-dir --target="/tmp/$SLURM_JOB_ID/pip" umap-learn || echo "[pip] umap-learn install failed; atlas_build.py will fall back to PCA"

PYTHONPATH="$FM:/tmp/$SLURM_JOB_ID/pip" python -u "$SC/atlas_build.py" \
  --emb "$BASE/merchant_embeddings.parquet" \
  --csv "$WORK/card_transaction.v1.csv" \
  --out "$BASE/atlas.json" \
  --copy-out "$BASE/scale/atlas/atlas.json" \
  --max-points 3000 --seed 42

echo "=== atlas ALL DONE $(date -u +%FT%TZ) ==="
ls -la "$BASE/atlas.json" "$BASE/scale/atlas/"
