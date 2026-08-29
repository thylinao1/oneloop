#!/bin/bash
#SBATCH --job-name=amex-safety-mia-pop
#SBATCH --partition=normal
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# SAFE-A, stage 1 of 3. CPU only, no GPU, no queue pressure on the GPU partitions.
# Recomputes the pretraining membership boundary from prep/{user.npy, ts.npy, meta.json}
# and HARD-FAILS if any of the six asserted population counts has drifted. Also picks the
# pre-registered rows-per-account m, so the GPU job's cost is known before it is submitted.
set -euo pipefail
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$XDG_CACHE_HOME"

source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
cd "$BASE/scripts/safety"

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
free -g | head -2 || true

python mia_eval.py --stage population --prep "$BASE/prep" --window 16 --seed 7
echo "=== POPULATION STAGE DONE $(date -u +%FT%TZ) ==="
