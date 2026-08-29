#!/bin/bash
#SBATCH --job-name=amex-ladder-readback
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --exclude=xgpj0
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# Closes the read-back owed on results/ladder.json. Job 751224 step 4 ran the ladder's own
# --check pass and its "CHECK OK" line was never read off the cluster because the VPN
# dropped mid-pass. This job re-runs that read-back from scratch, twice:
#   PASS A: the canonical command, byte-for-byte the step-4 invocation, unmodified script.
#   PASS B: the same invocation through ladder_readback.py, which calls the ladder's own
#           compare() for the verdict and additionally records the max observed difference.
# --check never writes: ladder_eval.main() returns from the compare branch before
# atomic_write_json, so ladder.json is read-only here. The sha256 is printed either side.
set -uo pipefail
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
mkdir -p "$XDG_CACHE_HOME"

source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
cd "$BASE/scripts/fm"

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
nproc
echo "--- sha256 BEFORE ---"
sha256sum ladder_eval.py ladder_readback.py "$BASE/ladder.json"

COMMON=(--stage run --prep "$BASE/prep" --eval "$BASE/eval" --asof "$BASE/eval/asof.npy"
        --user-emb "$BASE/eval/user_full_emb.npz"
        --merchants "$BASE/merchant_embeddings.parquet"
        --threads 24 --seed 7)

echo "=== PASS A: canonical --check (ladder_eval.py) $(date -u +%FT%TZ) ==="
python ladder_eval.py "${COMMON[@]}" --out "$BASE/ladder.json" \
  --shipped "$BASE/shipped_backbone_transfer.json" --bootstrap 1000 --check
RCA=$?
echo "=== PASS A DONE (exit $RCA) $(date -u +%FT%TZ) ==="

echo "=== PASS B: same read-back with max-diff instrumentation $(date -u +%FT%TZ) ==="
export LADDER_READBACK_REPORT=$BASE/logs/ladder-readback-$SLURM_JOB_ID.json
python ladder_readback.py "${COMMON[@]}" --out "$BASE/ladder.json" \
  --shipped "$BASE/shipped_backbone_transfer.json" --bootstrap 1000 --check
RCB=$?
echo "=== PASS B DONE (exit $RCB) $(date -u +%FT%TZ) ==="

echo "--- sha256 AFTER ---"
sha256sum ladder_eval.py ladder_readback.py "$BASE/ladder.json"
echo "--- report ---"
cat "$LADDER_READBACK_REPORT" 2>/dev/null || echo "no report written"

echo "=== READBACK ALL DONE (passA=$RCA passB=$RCB) $(date -u +%FT%TZ) ==="
test "$RCA" -eq 0 -a "$RCB" -eq 0
