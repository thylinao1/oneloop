#!/bin/bash
#SBATCH --job-name=amex-ladder-smoke2
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
python - <<'PY'
import numpy as np, os
u = np.load(os.path.expanduser("~/amex-oneloop/prep/user.npy"), mmap_mode="r")
n = int(u.max()) + 1
np.savez(os.environ["WORK"] + "/stub_user_emb.npz",
         emb=np.zeros((n, 512), dtype=np.float32), count=np.ones(n, dtype=np.int64))
print("stub embedding for", n, "accounts")
PY
python ladder_eval_v2.py --stage run --prep "$BASE/prep" --eval "$BASE/eval" \
  --asof "$BASE/eval/asof.npy" --user-emb "$WORK/stub_user_emb.npz" \
  --merchants "$BASE/merchant_embeddings.parquet" --threads 16 --seed 7 \
  --out "$WORK/stub2.json" --smoke 40000 --bootstrap 20 \
  --fraud-estimators 40 --mcc-estimators 20
python -c "
import json
d=json.load(open('$WORK/stub2.json'))
assert [r['rung'] for r in d['rungs']]==['L0','L1','L2','L3']
assert d['l0_shared_accounts']>0
gp=d['guard_prices_by_key']
paired=[k for k,v in gp.items() if v.get('paired')]
unpaired=[k for k,v in gp.items() if not v.get('paired')]
assert len(paired)==8 and len(unpaired)==4, (len(paired),len(unpaired))
for k in paired: assert len(gp[k]['price_ci'])==2, k
print('[stub2] OK; 8 paired price CIs, 4 unpaired steps')
print('  sample:', paired[0], gp[paired[0]]['price'], gp[paired[0]]['price_ci'])"
echo "=== STUB2 DONE $(date -u +%FT%TZ) ==="
