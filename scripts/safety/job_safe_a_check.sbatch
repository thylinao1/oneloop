#!/bin/bash
#SBATCH --job-name=amex-safety-mia-check
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclude=xgpj0,xgpe2
#SBATCH --constraint="cuda80|cuda90"
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# SAFE-A, the --check half. Recomputes every score, every metric and every interval from
# the same frozen inputs and compares every numeric leaf of the committed
# safety_membership.json at 1e-6. Exit 5 is a reproducibility finding and would ship as one.
#
# PIN THE NODE. Cross-node non-reproducibility is a measured effect in this repo, not a
# hypothesis: the ladder read-back reproduced at 1e-6 on the producing node and did not on
# a different CPU generation. Submit with:
#     sbatch --nodelist=<producing node> scripts/safety/job_safe_a_check.sbatch
set -euo pipefail
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$XDG_CACHE_HOME"

source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
cd "$BASE/scripts/safety"

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=name --format=csv || true
python -c "import torch,sys; cc=torch.cuda.get_device_capability(0); print('[guard] compute capability', cc); sys.exit(0 if cc >= (7,5) else 3)"
sha256sum "$BASE/safety_membership.json"

set +e
python mia_eval.py --stage run --prep "$BASE/prep" --ckpt "$BASE/ckpt/final.pt" \
  --out "$BASE/safety_membership.json" --window 16 \
  --device cuda --batch-size 1024 --threads 8 --seed 7 --bootstrap 1000 --check
RC=$?
set -e
echo "=== CHECK DONE (exit $RC) on $(hostname) $(date -u +%FT%TZ) ==="
sha256sum "$BASE/safety_membership.json"
echo "=== ALL DONE (check=$RC) ==="
