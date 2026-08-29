#!/bin/bash
#SBATCH --job-name=amex-protection-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=xgpj0
# job 751924 died on a TITAN V (compute capability 7.0): the cluster torch build
# ships no kernels below 7.5. Constrain to the GPU generations it does support.
#SBATCH --constraint="cuda75|cuda80|cuda90"
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# The --check half. Recomputes every score, every metric and every interval
# from the same frozen inputs and compares every numeric leaf of the committed
# protection.json at 1e-6. Exit 5 = a reproducibility finding, and it would ship as one.
set -euo pipefail
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$XDG_CACHE_HOME"

source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
cd "$BASE/scripts/fm"

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=name --format=csv || true
python -c "import torch,sys; cc=torch.cuda.get_device_capability(0); print('[guard] compute capability', cc); sys.exit(0 if cc >= (7,5) else 3)"

set +e
python protection_pll.py --stage run --prep "$BASE/prep" --eval "$BASE/eval" \
  --ckpt "$BASE/ckpt/final.pt" --out "$BASE/protection.json" \
  --device cuda --batch-size 1024 --threads 8 --seed 7 --bootstrap 1000 --check
RC=$?
set -e
echo "=== CHECK DONE (exit $RC) $(date -u +%FT%TZ) ==="
echo "=== ALL DONE (check=$RC) $(date -u +%FT%TZ) ==="
