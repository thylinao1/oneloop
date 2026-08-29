# Pre-registration F-C: resolving the mixed offer head on the real corpus (finals menu, second item)

Status: written on 2026-08-25 under the approved finals menu (2026-08-24). It became a
pre-registration when committed to main unchanged. Runs only on the shortlist
trigger, after F-D under the two-job budget. Results ship as obtained, in finals material only;
the uploaded Round 1 artifact is never modified.

## Question

The Complete Journey household offers head (post-cut coupon redemption, AUC delta on counts plus
demographics) is MIXED on the page: null in seed 7 (+0.0141 [-0.0094, 0.0382]) and positive in
seed 8 (+0.0269 [0.0023, 0.0551]) (results/cj/cj_offer_head_b7.json, cj_offer_head_b8.json).
Two further pretraining seeds make four. Does the mixed result resolve?

## Design, fixed now

- Two additional backbones on the Complete Journey corpus, seeds 9 and 10, with the committed
  adapter code unchanged (scripts/fm/cj_run.py installs the field list; pretrain.py and embed.py
  execute byte for byte as they did for seeds 7 and 8).
- The offer head exactly as shipped (scripts/fm/cj_offer_head.py), against the one pre-registered
  household-disjoint split from CJ-REPLICATION-PREREG.md and its amendment, household-clustered
  paired bootstrap B = 1000, the same counts-plus-demographics control.
- The store head and the transfer task are NOT re-run under this item; they are not the mixed
  result.

## Decision rule, frozen verbatim

With four seeds, the finals sentence calls the head positive only if at least three of four
intervals exclude zero in the positive direction; null if at most one does; otherwise it stays
mixed. All four values print whenever any is quoted. The standing rule is unchanged and binds every
sentence: the offer head is never quoted as one seed alone, and the Complete Journey corpus is
never described as card data.

What the count measures, said here and in whichever sentence the rule selects (fixed on
2026-08-24, before any run): the four seeds are pretraining seeds over one corpus, one recipe and one
pre-registered household split, so they are correlated by construction. Counting how many of four
intervals exclude zero is a replication rate across initialisations, meaning how reliably the
head reproduces when only the random start of the backbone changes. It is not four independent
samples of households and not a confidence statement about households; the household-clustered
interval inside each seed is the only statement about sampling here. The three-of-four rule is
kept because it is the shipped rule's natural extension, and the sentence it selects says
"replicates in [k] of four pretraining seeds" rather than anything that reads as pooled evidence.

## Corpus and provenance

Real grocery transactions from dunnhumby's Complete Journey, about 2,500 households, research use
terms, never card data. Campaigns in this corpus were targeted by the retailer and not randomized,
so nothing here estimates what an offer causes; the causal offers exhibit remains the Criteo
randomized result. The sentence carries that caveat where the shipped page carries it.

## Output and cost

results/cj/cj_offer_head_b9.json and cj_offer_head_b10.json, plus the backbone metadata files
the shipped runs produce. Two backbone runs on the small corpus (the original job covered two seeds,
so this doubles that job's shape) plus two head evaluations. Exact accounting from Slurm after the
run.

## Paired sentences, drafted before the run

Positive: "The offer head was mixed on two seeds, so we ran two more in the finals week under a
rule written before we knew we would be here. It replicates in [k] of four pretraining seeds,
which is a replication rate across initialisations on one corpus and one split, not four
independent samples, and all four intervals are printed."

Still mixed: "The offer head was mixed on two seeds, so we ran two more. It replicates in two of
four pretraining seeds, which is still mixed, and we say mixed."

Null: "The offer head was mixed on two seeds, so we ran two more. It replicates in [k] of four
pretraining seeds, which by the rule we wrote first is a null, and the store head, positive on
both seeds, is the merchant-level result that stands."
