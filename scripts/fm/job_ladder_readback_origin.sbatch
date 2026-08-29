#!/bin/bash
#SBATCH --job-name=amex-ladder-readback-origin
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --nodelist=xcnc20
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# Same read-back as job 751710, pinned to xcnc20, the node that produced results/ladder.json
# and ran the original in-job --check. Tests whether the 26 differing leaves seen on xcnf27
# are node-dependent floating point or a real non-reproduction of the stored numbers.
set -uo pipefail
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
mkdir -p "$XDG_CACHE_HOME"
source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
cd "$BASE/scripts/fm"
echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
sha256sum ladder_eval.py ladder_readback.py "$BASE/ladder.json"
grep -m1 "model name" /proc/cpuinfo || true
export LADDER_READBACK_REPORT=$BASE/logs/ladder-readback-$SLURM_JOB_ID.json
python ladder_readback.py --stage run --prep "$BASE/prep" --eval "$BASE/eval" \
  --asof "$BASE/eval/asof.npy" --user-emb "$BASE/eval/user_full_emb.npz" \
  --merchants "$BASE/merchant_embeddings.parquet" --threads 24 --seed 7 \
  --out "$BASE/ladder.json" --shipped "$BASE/shipped_backbone_transfer.json" \
  --bootstrap 1000 --check
RC=$?
echo "=== ORIGIN READBACK DONE (exit $RC) $(date -u +%FT%TZ) ==="
sha256sum "$BASE/ladder.json"
cat "$LADDER_READBACK_REPORT" 2>/dev/null || echo "no report"
