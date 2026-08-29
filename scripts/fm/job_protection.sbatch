#!/bin/bash
#SBATCH --job-name=amex-protection
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
# PROTECTION, made measurable. EVALUATION ONLY against the frozen
# checkpoint: masked-field pseudo-log-likelihood as a LABEL-FREE surprise score,
# measured on the same frozen evaluation split the shipped transfer table uses,
# against a no-model unigram rarity control.
#   1) code-path smoke (small row count, fail fast)
#   2) the full pass -> protection.json
# The --check reproduction runs as its own job (job_protection_check.sbatch) so
# neither half can run into the 3h wall of the gpu partition.
set -euo pipefail
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$XDG_CACHE_HOME"

source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
WORK=/tmp/$SLURM_JOB_ID/work
mkdir -p "$WORK"
cd "$BASE/scripts/fm"

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
# fail fast and loudly rather than burning the allocation on an unusable GPU
python -c "import torch,sys; cc=torch.cuda.get_device_capability(0); print('[guard] compute capability', cc); sys.exit(0 if cc >= (7,5) else 3)"
free -g | head -2 || true

COMMON=(--stage run --prep "$BASE/prep" --eval "$BASE/eval" --ckpt "$BASE/ckpt/final.pt"
        --device cuda --batch-size 1024 --threads 8 --seed 7)

echo "=== STEP 1: code-path smoke $(date -u +%FT%TZ) ==="
python protection_pll.py "${COMMON[@]}" --out "$WORK/smoke_protection.json" \
  --smoke 20000 --bootstrap 30
python -c "
import json
d=json.load(open('$WORK/smoke_protection.json'))
assert d['zero_labels']['fraud_labels_in_score_construction'] is False
assert d['zero_labels']['score_sha256_before_labels'] == d['zero_labels']['score_sha256_at_report']
assert set(d['scores']) >= {'pll_all_fields','rarity_all_fields','pll_behavioural_only'}
print('[smoke] scores present, zero-label guard held, sha', d['zero_labels']['score_sha256_before_labels'][:12])"
echo "=== SMOKE PASSED $(date -u +%FT%TZ) ==="

echo "=== STEP 2: full pass, 300k scored rows, B=1000 $(date -u +%FT%TZ) ==="
python protection_pll.py "${COMMON[@]}" --out "$BASE/protection.json" --bootstrap 1000
echo "=== STEP 2 DONE $(date -u +%FT%TZ) ==="
python -c "
import json
d=json.load(open('$BASE/protection.json'))
for k,v in d['scores'].items():
    print(f\"{k:34s} roc {v['roc_auc']:.6f} {[round(x,6) for x in v['roc_auc_ci']]} pr {v['pr_auc']:.6f} {[round(x,6) for x in v['pr_auc_ci']]}\")
for k,e in d['comparisons_by_key'].items():
    print(k, '| pr', round(e['pr_auc']['difference'],6), [round(x,6) for x in e['pr_auc']['ci']], e['pr_auc']['direction'],
          '| roc', round(e['roc_auc']['difference'],6), [round(x,6) for x in e['roc_auc']['ci']], e['roc_auc']['direction'])
print('prevalence', d['split']['positive_rate'], 'positives', d['split']['n_positives'])
print({k:v for k,v in d['diagnostics'].items() if 'roc_auc_of' in k})"

echo "=== ALL DONE $(date -u +%FT%TZ) ==="
ls -la "$BASE/protection.json"
