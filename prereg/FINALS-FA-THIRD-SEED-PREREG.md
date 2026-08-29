# Pre-registration F-A: a third pretraining seed on the synthetic corpus (finals menu)

Status: written on 2026-08-25 under the approved finals menu (2026-08-24). It became a
pre-registration when committed to main unchanged. After that commit nothing below
changes. Runs only on the shortlist trigger, after F-D and F-C under the two-job budget. Results
ship as obtained, in finals material only; the uploaded Round 1 artifact is never modified.

## Question

The shipped L3 next-category transfer gain is +0.0218 top-1 on seed 7 [0.0103, 0.0343] and holds
on seed 1337 [0.0110, 0.0334] (results/backbone_transfer.json, scaling_by_tag scale-24m-seed2).
Does it survive a third pretraining seed?

## Design, fixed now

- One backbone at seed 9, the shipped recipe unchanged (scripts/fm/pretrain.py, d_model 512,
  6 layers, 8 heads, window 16, 8 epochs, the same corpus cut and the same prep arrays).
- The existing L3 transfer evaluation unchanged (scripts/fm/transfer_eval.py), the same
  pre-registered split, entity-clustered paired bootstrap B = 1000, seed 7 for the split as
  shipped.
- Metrics reported: next-category top-1 and top-5 delta, fraud AUC and PR-AUC delta, all with
  intervals, as the shipped table reports them.

## Abort criterion, fixed blind (2026-08-24)

A negative result and a broken run must not print the same sentence. The seed-9 run counts as a
seed result only if all of the following hold, checked by the producing script before the
transfer evaluation is read:

1. It completes all 8 epochs on the same corpus cut (2017-08-25T09:37:00Z) and the same prep
   arrays (the prep directory's meta.json sha256 recorded in the run's metadata must equal the
   one the shipped seed-7 run records), reaching the same optimizer step count, 78,040, that both
   reference runs reached.
2. Its final recorded training loss lands inside the band [0.5058, 0.5458]. The band is the mean
   of the two reference final losses plus or minus 0.02: the shipped seed-7 run ended at 0.52603
   at step 78,040 (results/backbone_transfer.json, pretrain.loss_curve, last point) and the
   seed-1337 repeat ended at 0.52555 at step 78,040
   (results/scale/scale-24m-seed2.json, pretrain.loss_curve, last point), so the two references
   differ by 0.00048, and seed 7's whole final epoch ranges 0.51961 to 0.53086. Plus or minus
   0.02 is about four times the within-epoch spread and about forty times the seed-to-seed gap,
   wide enough that an honest seed cannot fail it and narrow enough that a diverged, truncated
   or wrongly fed run cannot pass it.
3. The checkpoint loads and the embedding pass produces vectors for every held-out account, with
   0 unmatched rows, as the shipped evaluation asserts.

Outside that, the run is reported as a FAILED RUN, in those words, with the reason (which of the
three checks failed and the observed value), and it is never reported as a seed. A failed run may
be re-submitted once under this same file; a second failure closes the item as "the third seed
did not complete" and no transfer number from it is ever quoted.

## Decision rule, frozen verbatim

The finals sentence is "the gain held on all three seeds" only if the seed-9 top-1 interval
excludes zero in the positive direction; otherwise the sentence is "the gain held on two of three
seeds and the third came back [value as obtained]", printed at the same prominence. No seed
retroactively upgrades a shipped claim: the page's +0.0218 stays the shipped number, and the third
seed adds "held on a third seed" or "did not hold on a third seed" and nothing else. All three
seeds' values print whenever any one is quoted.

## Corpus and provenance

Synthetic IBM TabFormer, labelled synthetic in every sentence. Seeds are correlated by
construction (same corpus, same recipe), so this tests seed sensitivity and never sampling
variability, and the sentence says so if asked.

## Output and cost

results/scale/scale-24m-seed9.json merged by scripts/merge_scaling.py under tag
`scale-24m-seed9` (the merge script refuses a duplicate tag). One pretrain, about the shipped
main run's GPU-hours on an A100 (results/cost_model.json records the shipped a100-40 hours), plus
one evaluation pass. Exact accounting from Slurm after the run.

## Paired sentences, drafted before the run

Held: "Two seeds is not many, so we ran a third in the finals week under a rule written before we
knew we would be here. It held."

Did not hold: "Two seeds is not many, so we ran a third in the finals week under a rule written
before we knew we would be here. It did not hold, and the page's number stays what it was with
that sentence beside it."
