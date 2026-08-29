# CJ-RUNBOOK: Complete Journey replication sequence

Cluster: `ssh soc` (needs NUS VPN; if it times out the VPN dropped). All paths below are
on the cluster unless marked LOCAL. Master cluster guide: `~/Developer/NUS-COMPUTE.md`
(LOCAL).

Hard constraints, every step:
- max 2 concurrent own jobs on this account
- NEVER touch `tta-*` jobs; never wildcard `scancel`; cancel only by explicit own job id
- commit CJ-REPLICATION-PREREG.md alone BEFORE step 3
- results ship as obtained; no resubmits to chase a number

## 0. Stage code and data

LOCAL, from the repo root (code to the cluster's scripts/fm; the fm/ originals are
already there from the TabFormer runs, this adds the cj_* files):

    rsync -av scripts/fm/ soc:~/amex-oneloop/scripts/fm/

Data: the verified zip is already staged at `~/amex-oneloop/cj/source.zip`
(zip sha256 5e0a3d72fe8562fe0ab995f70fb58b74359e8ec4bbccd1521e2b137da0558f9a). If it is
ever missing, re-stage it from the verified local copy:

    # LOCAL
    scp /path/to/dunnhumby_The-Complete-Journey.zip soc:~/amex-oneloop/cj/source.zip

No Kaggle credentials are involved (the Kaggle mirror is not used; no redistribution
right).

## 1. Download/verify job (CPU, ~minutes)

    sbatch ~/amex-oneloop/scripts/fm/job_cj_download.sbatch

If the zip were absent you could pass a refreshed direct link instead:

    sbatch --export=ALL,DATA_URL='https://...' ~/amex-oneloop/scripts/fm/job_cj_download.sbatch

## 2. Verify provenance before anything trains

    cat ~/amex-oneloop/cj/meta.json
    cat ~/amex-oneloop/cj/CHECKSUMS.txt

Gate: `files.transaction_data.csv.rows_observed == 2595732`, product 92353,
hh_demographic 801, coupon_redempt 2318, and the four `.parquet` files exist. The job
exits nonzero on any verified-count or column mismatch; do not work around it.

## 3. Prereg commit (local)

Commit `CJ-REPLICATION-PREREG.md` alone. Only after that commit exists:

## 4. Backbone job (GPU a100-40, both pretrain seeds + all embeddings in one job)

    sbatch ~/amex-oneloop/scripts/fm/job_cj_backbone.sbatch

Runtime expectation, derived not promised: the TabFormer backbone took 1h42m for 8
epochs on 24M rows (a100). Complete Journey has 2.6M rows (about 2.1M pre-cut), so per
seed roughly 24M/2.6M ~ 9x fewer optimizer steps: on the order of 10-15 minutes per
seed, under an hour for both seeds plus prep, smoke and the three embedding passes. The
6h limit is headroom. Resubmit-safe: pretrain resumes from latest.pt, prep skips if done.

Watch (own jobs only):

    squeue --me
    tail -f ~/amex-oneloop/logs/amex-cj-backbone-<jobid>.out

## 5. Transfer + heads job (CPU, after step 4 completes)

    sbatch ~/amex-oneloop/scripts/fm/job_cj_transfer.sbatch

Produces in `~/amex-oneloop/cj/`: `cj_transfer_s7.json`, `cj_transfer_s8.json`,
`cj_transfer.json` (combined, carries the required sentence), `cj_store_head_b7.json`,
`cj_store_head_b8.json`, `cj_offer_head_b7.json`, `cj_offer_head_b8.json`. The job ends
with --check reruns of both heads; a nonzero exit means a results file did not reproduce
and must not be pulled.

Steps 4 and 5 are sequential (5 gates on 4's outputs), so the 2-job budget is never
exceeded: at most the one running job plus one queued.

## 6. Pull results (LOCAL)

    rsync -av soc:~/amex-oneloop/cj/cj_*.json results/cj/
    rsync -av soc:~/amex-oneloop/cj/meta.json results/cj/cj_download_meta.json

Then read `cj_transfer.json.required_sentence` and the two heads' `required_sentence`
fields. Those sentences, whatever they say, are the finding. The pre-registration
(FIX-LIMITS-PLAN.md) decides separately whether anything lands in the artifact; nothing
in this runbook touches the frozen `4877aad` artifact.

## Notes for the eval-code reader

transfer_eval.py could not be reused unchanged for the task itself: it hard-codes the
("fraud", "next_mcc") task pair, TabFormer field names inside its feature builder, and
binds common.FIELDS at import time. transfer_eval_cj.py therefore exists, but it imports
paired_cluster_bootstrap and VAL_USER_FRAC from transfer_eval.py, so the interval
machinery and validation-split convention are the shipped code objects. pretrain.py and
embed.py DO run unmodified (cj_run.py installs the field list into common first).

Two protocol points specific to this corpus, both asserted in code and covered by the
prereg amendment: (1) the scored unit is the BASKET-OPENING transaction, because rows
of one basket share a timestamp and a mid-basket row would hand embed.py's as-of window
its own basket's items; history features aggregate over strictly-earlier ts for both
arms, and the eval pool is roughly one row per post-cut basket. (2) the run-stage eval
seed is pinned at 42 for both backbone seeds, so seed-to-seed differences in
cj_transfer.json are pretraining variability, not evaluation noise.
