#!/bin/bash
#SBATCH --job-name=amex-ladder-smoke
#SBATCH --partition=long
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
set -euo pipefail
source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
export WORK=/tmp/$SLURM_JOB_ID/work; mkdir -p "$WORK"
cd "$BASE/scripts/fm"
echo "=== node $(hostname) | $(date -u +%FT%TZ) ==="
# STUB account embedding (zeros) purely to exercise the run-stage code path.
# Never used for a reported number; the real pooled embedding is job 751224 step 1.
python - <<'PY'
import numpy as np, os
u = np.load(os.path.expanduser("~/amex-oneloop/prep/user.npy"), mmap_mode="r")
n = int(u.max()) + 1
np.savez(os.environ["WORK"] + "/stub_user_emb.npz",
         emb=np.zeros((n, 512), dtype=np.float32), count=np.ones(n, dtype=np.int64))
print("stub embedding for", n, "accounts")
PY
python ladder_eval.py --stage run --prep "$BASE/prep" --eval "$BASE/eval" \
  --asof "$BASE/eval/asof.npy" --user-emb "$WORK/stub_user_emb.npz" \
  --merchants "$BASE/merchant_embeddings.parquet" --threads 16 --seed 7 \
  --out "$WORK/stub_ladder.json" --smoke 40000 --bootstrap 20 \
  --fraud-estimators 40 --mcc-estimators 20
python -c "
import json
d=json.load(open('$WORK/stub_ladder.json'))
assert [r['rung'] for r in d['rungs']]==['L0','L1','L2','L3']
assert d['l0_shared_accounts']>0
print('[stub-smoke] OK; L0 shares', d['l0_shared_accounts'], 'accounts')"
echo "=== STUB SMOKE DONE $(date -u +%FT%TZ) ==="
