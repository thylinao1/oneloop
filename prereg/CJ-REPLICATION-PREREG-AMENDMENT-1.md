# Amendment 1 to CJ-REPLICATION-PREREG.md

Status: written and committed BEFORE any training or evaluation run. At amendment time the
only cluster activity had been the data staging and one download-verify attempt (job 752750)
that failed at a python import error before performing any verification or conversion. No
prep output, no checkpoint, no evaluation number of any kind existed.

Reason: a four-reviewer adversarial pass on the adapter code, run after the prereg commit
6c9c7a9 and before any run, found defects that would have violated the prereg's own stated
protocol. Fixing them changes implementation details the prereg described, so the changes
are recorded here rather than silently. All decision rules, hypotheses, arms, metrics,
thresholds and seeds in the prereg stand unchanged.

## A1. Scored unit for H1 is the basket-opening transaction

The prereg registered "as-of embeddings pooled strictly before the scored transaction".
On this corpus every item of a basket shares one timestamp, so a row-index window honors
that wording by row order but not by time: at a mid-basket row the window is filled mostly
with the same shopping trip's other items, and the baseline arm sees only one prior item.
The delta would have measured within-basket co-occurrence, not transfer.

Fix, keeping embed.py byte-identical: only basket-opening rows are scored (first row of a
post-cut basket whose predecessor row has strictly smaller timestamp). Basket starts that
tie an earlier basket on the same timestamp are excluded and counted in the output
(n_basket_starts_dropped_same_ts_tie). The evaluation pool therefore shrinks to about one
row per post-cut basket. An independent verifier in the smoke rebuilds every selected
row's window and confirms no same-timestamp and no cross-household row enters any window.

## A2. History features are strictly-earlier-time aggregates in both arms

The baseline's history features previously used a one-row shift, which inside a basket
reads a same-timestamp item. They are rebuilt on time-groups per household, shifted by one
group, so both arms condition on strictly earlier timestamps only. Feature names and
their roles are unchanged.

## A3. Evaluation seed pinned

The run stage previously received the backbone seed, which also drove the validation
split, the LightGBM seed and the bootstrap draws, conflating pretraining variability with
evaluation noise. The evaluation seed is pinned to 42 for both backbone seeds; the
backbone seed enters only through checkpoint and embedding paths, and the combine stage
asserts the two inputs used the same evaluation seed. The H1 decision rule (both top-1
intervals above zero) is unchanged.

## A4. Offer-head unscorable floor made explicit

The prereg said too-few positives ship as the finding. The floor is now concrete: fewer
than 5 test-window positive households on either label side, or a degenerate bootstrap
(all resamples single-class), produces the honest unscorable JSON with point estimates
and no interval claim. The H3 decision rule is unchanged for scorable runs.

## A5. Defect fixes with no protocol effect, listed for completeness

Wrong Slurm partition on the backbone job (gpu to gpu-long); numpy missing from the
download job's environment (the cause of the failed job 752750); household total reported
from max-plus-one on raw 1-based keys (2,501 for 2,500) corrected to the prep metadata
count; a null guard on RETAIL_DISC before quantile edges; a store-head assertion that the
embedding file's per-store pre-cut counts equal counts recomputed from the prep arrays;
exclusion accounting decomposed (post-cut-only, missing store key, below floor); download
resumability (partial file cannot poison the checksum gate); one known bad node excluded
from the transfer job; the user-guide citation made conditional on the file's presence.
