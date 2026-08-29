#!/bin/bash
#SBATCH --job-name=amex-safety-mia
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclude=xgpj0,xgpe2
# job 751924 died on a TITAN V (compute capability 7.0): the cluster torch build ships no
# kernels below 7.5. Constrain to the GPU generations it does support, and guard again at
# runtime so an unusable GPU costs seconds rather than the whole allocation.
#SBATCH --constraint="cuda80|cuda90"
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# SAFE-A. Membership inference against the frozen backbone. EVALUATION ONLY: the
# checkpoint is read for inference and nothing is retrained. prep/fraud.npy is never
# opened, so the run is label-free by construction.
#   1) population stage, asserts the membership counts before any GPU time is spent
#   2) code-path smoke, small row count, fail fast
#   3) the full pass -> safety_membership.json
# The --check reproduction runs as its own job, pinned to THIS node.
set -euo pipefail
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$XDG_CACHE_HOME"

source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
WORK=/tmp/$SLURM_JOB_ID/work
mkdir -p "$WORK"
cd "$BASE/scripts/safety"

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
echo "=== PRODUCING NODE: $(hostname) (pin the --check job here with --nodelist) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch,sys; cc=torch.cuda.get_device_capability(0); print('[guard] compute capability', cc); sys.exit(0 if cc >= (7,5) else 3)"
free -g | head -2 || true

COMMON=(--prep "$BASE/prep" --ckpt "$BASE/ckpt/final.pt" --window 16
        --device cuda --batch-size 1024 --threads 8 --seed 7)

echo "=== STEP 1: population + membership asserts $(date -u +%FT%TZ) ==="
python mia_eval.py --stage population --prep "$BASE/prep" --window 16 --seed 7

echo "=== STEP 2: code-path smoke $(date -u +%FT%TZ) ==="
python mia_eval.py --stage run "${COMMON[@]}" --out "$WORK/smoke_mia.json" \
  --smoke 20000 --bootstrap 20
python -c "
import json
d=json.load(open('$WORK/smoke_mia.json'))
assert d['zero_labels']['fraud_labels_opened'] is False
assert d['population']['all_scored_rows_post_cut'] is True
p=d['strata'][d['population']['stratum_primary']]['arms']
for k in ('model_pll_behavioural','control_unigram_rarity','calibration_positive_tenure',
          'calibration_negative_random'):
    assert k in p, k
    print(f\"{k:36s} row_auc {p[k]['row_auc']:.4f}  acct_auc {p[k]['account_auc']:.4f}\")
print('[smoke] arms present, label-free guard held, all scored rows post-cut')"
echo "=== SMOKE PASSED $(date -u +%FT%TZ) ==="

echo "=== STEP 3: full pass, B=1000 $(date -u +%FT%TZ) ==="
python mia_eval.py --stage run "${COMMON[@]}" --out "$BASE/safety_membership.json" \
  --bootstrap 1000
echo "=== STEP 3 DONE $(date -u +%FT%TZ) ==="

python -c "
import json
d=json.load(open('$BASE/safety_membership.json'))
pr=d['population']['stratum_primary']
print('primary stratum:', pr, '| m =', d['population']['m_chosen'])
st=d['strata'][pr]
print('member rows', st['n_member_rows'], 'accounts', st['n_member_accounts'],
      '| non-member rows', st['n_non_member_rows'], 'accounts', st['n_non_member_accounts'])
print('row FPR floor', st['fpr_floor']['row_level']['smallest_measurable_fpr'],
      '| account FPR floor', st['fpr_floor']['account_level']['smallest_measurable_fpr'])
for k,v in st['arms'].items():
    print(f\"{k:36s} row_auc {v['row_auc']:.4f} {[round(x,4) for x in (v['row_auc_ci'] or [float('nan')]*2)]}\"
          f\"  tpr@1e-3 {v['row_tpr_at_fpr_0.001']:.5f}  tpr@1e-4 {v['row_tpr_at_fpr_0.0001']:.5f}\"
          f\"  acct_auc {v['account_auc']:.4f} {[round(x,4) for x in (v['account_auc_ci'] or [float('nan')]*2)]}\")
for k,e in d['comparisons_by_key'].items():
    print(k, '| row', round(e['row_auc']['difference'],5), e['row_auc']['ci'], e['row_auc']['direction'],
          '| acct', round(e['account_auc']['difference'],5), e['account_auc']['ci'], e['account_auc']['direction'])
print('MDE:', {k: (round(v['account_auc'],4), v['account_auc_ci'] and [round(x,4) for x in v['account_auc_ci']]) for k,v in st['minimum_detectable_effect']['by_delta'].items()})
print('gradient:', {k: (v['n_member_rows'], v['row_auc'] and round(v['row_auc'],4), v['control_row_auc'] and round(v['control_row_auc'],4)) for k,v in d['diagnostics_not_attacks']['contamination_gradient_by_window'].items()})
print('tenure:', d['diagnostics_not_attacks']['account_tenure']['median_total_rows_member_accounts'],
      d['diagnostics_not_attacks']['account_tenure']['median_total_rows_non_member_accounts'])"

echo "=== ALL DONE on $(hostname) $(date -u +%FT%TZ) ==="
ls -la "$BASE/safety_membership.json"
sha256sum "$BASE/safety_membership.json"
