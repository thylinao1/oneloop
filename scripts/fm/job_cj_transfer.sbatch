#!/bin/bash
#SBATCH --job-name=amex-cj-transfer
#SBATCH --partition=long
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --exclude=xgpj0
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/amex-oneloop/logs/%x-%j.out
#
# Complete Journey L3 transfer + the two heads. CPU only: LightGBM fits and bootstraps
# (the forward passes already happened in job_cj_backbone.sbatch). Per pre-registration:
# two backbone seeds against ONE pre-registered household-disjoint split, with the EVAL
# seed pinned at 42 for both transfer runs (the backbone seed enters only through the
# asof/ckpt paths, so seed-to-seed differences are pretraining variability). Then the
# store head and the offer head per seed, then --check reruns of the four HEAD files
# (the transfer JSONs have no check mode; they are reproduced by rerunning the stage).
set -euo pipefail
export XDG_CACHE_HOME=/tmp/$SLURM_JOB_ID/xdg
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
mkdir -p "$XDG_CACHE_HOME"

source "$HOME/amex-oneloop/venv/bin/activate"
BASE=$HOME/amex-oneloop
CJ=$BASE/cj
cd "$BASE/scripts/fm"

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date -u +%FT%TZ) ==="
free -g | head -2 || true
nproc

# input gate: backbone job must have finished for both seeds, prep must exist
test -f "$CJ/prep/meta.json" || { echo "FATAL: $CJ/prep/meta.json missing - run job_cj_backbone.sbatch first"; exit 2; }
for SEED in 7 8; do
  for f in "$CJ/ckpt/s$SEED/final.pt" "$CJ/ckpt/s$SEED/pretrain_summary.json" \
           "$CJ/eval/asof_s$SEED.npy" "$CJ/store_embeddings_s$SEED.parquet" \
           "$CJ/hh_emb_s$SEED.npz"; do
    test -f "$f" || { echo "FATAL: $f missing - run job_cj_backbone.sbatch first"; exit 2; }
  done
done
test -f "$CJ/coupon_redempt.parquet" || { echo "FATAL: coupon_redempt.parquet missing"; exit 2; }
test -f "$CJ/hh_demographic.parquet" || { echo "FATAL: hh_demographic.parquet missing"; exit 2; }

# EVAL seed pinned at 42 for both runs (pre-registered); only the backbone paths vary
for SEED in 7 8; do
  echo "=== L3 transfer, backbone seed $SEED $(date -u +%FT%TZ) ==="
  python transfer_eval_cj.py --stage run --prep "$CJ/prep" --eval "$CJ/eval" \
    --asof "$CJ/eval/asof_s$SEED.npy" \
    --ckpt-summary "$CJ/ckpt/s$SEED/pretrain_summary.json" \
    --out "$CJ/cj_transfer_s$SEED.json" --bootstrap 1000 --threads 24 --seed 42
done

echo "=== combine (two seeds -> one file with the required sentence) ==="
python transfer_eval_cj.py --stage combine \
  --inputs "$CJ/cj_transfer_s7.json" "$CJ/cj_transfer_s8.json" \
  --out "$CJ/cj_transfer.json"

for SEED in 7 8; do
  echo "=== store head, backbone seed $SEED $(date -u +%FT%TZ) ==="
  python cj_store_head.py --prep "$CJ/prep" \
    --emb "$CJ/store_embeddings_s$SEED.parquet" \
    --out "$CJ/cj_store_head_b$SEED.json" --backbone-seed "$SEED" --threads 4
  echo "=== offer head, backbone seed $SEED $(date -u +%FT%TZ) ==="
  python cj_offer_head.py --stage run --prep "$CJ/prep" \
    --emb "$CJ/hh_emb_s$SEED.npz" \
    --redemptions "$CJ/coupon_redempt.parquet" \
    --demographics "$CJ/hh_demographic.parquet" \
    --out "$CJ/cj_offer_head_b$SEED.json" --backbone-seed "$SEED" --threads 4
done

echo "=== --check reruns: every head file must reproduce at 1e-6 ==="
RC=0
for SEED in 7 8; do
  python cj_store_head.py --prep "$CJ/prep" \
    --emb "$CJ/store_embeddings_s$SEED.parquet" \
    --out "$CJ/cj_store_head_b$SEED.json" --backbone-seed "$SEED" --threads 4 --check \
    || RC=$?
  python cj_offer_head.py --stage run --prep "$CJ/prep" \
    --emb "$CJ/hh_emb_s$SEED.npz" \
    --redemptions "$CJ/coupon_redempt.parquet" \
    --demographics "$CJ/hh_demographic.parquet" \
    --out "$CJ/cj_offer_head_b$SEED.json" --backbone-seed "$SEED" --threads 4 --check \
    || RC=$?
done
echo "=== check pass exit code: $RC (0 = all reproduced) ==="

echo "=== ALL DONE $(date -u +%FT%TZ) ==="
ls -la "$CJ"/cj_*.json
exit "$RC"
