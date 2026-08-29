# Pre-registration: Complete Journey replication (fixes L-A and L-B)

Status: written before any run. This file is committed alone before
any training job is submitted. After that commit, nothing below changes: no added seeds,
no swapped outcomes, no threshold moved after seeing a number. Whatever the runs show
ships, including nulls and negatives.

Corpus decision trail: Elo (Kaggle) was the first choice and was dropped on license
grounds before any Elo code landed (competition-only use plus a delete-after obligation;
see scripts/fm/ELO-DROPPED.md). The replication corpus is dunnhumby "The Complete
Journey": real retail transactions from ~2,500 households over 711 days, released for
use "solely for research, personal or non-commercial purposes" (dunnhumby terms,
retrieved 2026-08-24).

## 1. What is being replicated

The shipped submission pretrains a masked-field transformer (d_model 512, 6 layers,
~20.5M params) on the synthetic IBM TabFormer corpus and evaluates transfer under the L3
protocol: entity-disjoint split, as-of embeddings pooled strictly before the scored
transaction, and a gradient-boosted baseline that also gets per-entity temporal
aggregates. Two stated limitations are in scope here:

- L-A: everything rests on synthetic data. Fix: replicate pretraining plus the L3
  transfer evaluation on a real corpus.
- L-B: no head consumes the backbone at merchant level against a real outcome. Fix: a
  store-level head and a household-level offers head, both reading backbone embeddings
  through keys the vocabulary never saw.

The same pretrain.py and embed.py execute byte for byte (cj_run.py installs the
Complete Journey field list into the shared common module before running them; no fm/
file is modified). The bootstrap interval code is imported from the shipped
transfer_eval.py.

## 2. Scale caveat, stated plainly

Complete Journey has 2.6M transactions and ~2,500 households against TabFormer's 24M
rows and ~2,000 accounts with far longer vocabularies. This is a protocol replication on
a small real corpus, not a scale replication. Entity-disjoint splits over 2,500
households are coarse: the split is about 2,000 train households and about 500 test
households, and every interval is household-clustered because of it. The store head has
a few hundred stores. Wide intervals are expected and will be reported as obtained.

## 3. Hypotheses and decision rules

### H1: L3 transfer gain on real data (fixes L-A)

Task: next COMMODITY_DESC (top-30 classes + other), the analog of the shipped next-MCC
task. Protocol: L3 only. Two pretraining seeds (7 and 8) are evaluated against ONE
pre-registered household-disjoint split (select stage, seed 7). Metric: top-1 and top-5
delta (with_emb minus baseline), 95% household-clustered paired bootstrap interval,
B = 1000.

Decision rule, fixed now: the result is called positive only if BOTH seeds' top-1
intervals sit entirely above zero. Both below zero is called negative. Anything else is
a null or mixed result. All three outcomes ship with the same prominence. The
majority-class floor is reported next to the deltas (an addition over the shipped
script, which does not report it).

### H2: store head, embeddings vs counts (fixes L-B at merchant level)

Unit: store. Outcome, chosen now between the two candidates: post-cut sales GROWTH
tercile, defined as (post-cut SALES_VALUE per post day) / (pre-cut SALES_VALUE per pre
day), cut at terciles over eligible stores (>= 50 pre-cut rows and positive pre-cut
sales). Growth rather than raw post-cut volume, because raw volume is mostly pre-cut
size restated and the counts control would win by construction without answering the
embedding question.

Arms: (a) counts-only control: log1p pre-cut transaction count, log1p distinct
households, mean basket value; (b) control plus the backbone's pre-cut pooled store
embedding. Store-disjoint split, 80/20, seed 42. Metrics: macro-F1 and accuracy;
the majority-tercile floor reported.

Decision rule, fixed now: embeddings are said to add signal only if the macro-F1 delta's
95% interval sits above zero. An interval spanning zero ships as a null with the
sentence already written into cj_store_head.py. The honest question is whether
embeddings add anything over counts; a null is an acceptable answer.

### H3: offer head, embeddings vs counts plus demographics (fixes L-B at household level)

Unit: household. Label: at least one coupon_redempt row with DAY at or after the corpus
cut. PREDICTIVE, NOT CAUSAL, and labeled so in the output: campaigns were targeted by
the retailer, not randomized, so nothing here estimates what an offer causes. The causal
offers exhibit in this project remains the Criteo randomized-uplift result.

Arms: (a) control: pre-cut counts (transactions, baskets, total sales, distinct stores,
coupon-discount rows, prior redemptions) plus the anonymized demographic attributes of
the 2023 re-release (classification_1..5, HOMEOWNER_DESC, KID_CATEGORY_DESC; 801 of the
~2,500 households have them, the rest carry NA codes); (b) control plus the pre-cut
pooled household embedding. Household-disjoint split, 80/20, seed 42. Metrics: AUC and
PR-AUC with 95% household-clustered intervals on the deltas.

Decision rule, fixed now: same shape as H2, judged on the AUC delta interval. Positive
counts in the test window are reported; if the test window has too few positives to
score, that is reported as the finding rather than tuned around.

## 4. Protocol constants, fixed in advance

- prep: post_frac 0.18 (the shipped default), vocabularies and quantile edges from
  pre-cut rows only, cut on DAY (recorded as cut_day in meta.json)
- fields: dow, hour, week (4-week buckets), amount_q (100 buckets), quantity (clipped),
  department, commodity (top-300 then UNK), brand, retail_disc_q (20 buckets)
- excluded from the vocabulary, with reasons recorded in meta.json: household_key,
  BASKET_ID, PRODUCT_ID, STORE_ID (identifiers; the store key exists only as the pooling
  index), COUPON_DISC and COUPON_MATCH_DISC (the redemption outcome family H3 predicts).
  RETAIL_DISC is an input: a shelf markdown observable at transaction time, not an outcome.
- backbone: d_model 512, 6 layers, 8 heads, ff 2048, window 16, stride 8, batch 256,
  8 epochs, seeds 7 and 8; architecture identical to the shipped backbone even though it
  is oversized for 2.6M rows, because the claim under test is the protocol, not a tuned model
- evaluation: caps 800k/300k (they will not bind here), VAL_USER_FRAC 0.1,
  LightGBM hyperparameters identical to the shipped next-MCC task, B = 1000
- heads: seeds 42, deterministic LightGBM settings, --check reruns must reproduce every
  numeric leaf at 1e-6

## 5. Provenance gate (before any training run)

No training job is submitted until the download job has:

1. verified the source zip against sha256
   5e0a3d72fe8562fe0ab995f70fb58b74359e8ec4bbccd1521e2b137da0558f9a (official dunnhumby
   CDN asset, staged 2026-08-24; source page https://www.dunnhumby.com/source-files/)
2. verified row counts: transaction_data.csv 2,595,732; product.csv 92,353;
   hh_demographic.csv 801; coupon_redempt.csv 2,318 (all verified manually 2026-08-24;
   mismatch is fatal). Counts for campaign_table, campaign_desc, coupon and causal_data
   are recorded as observed.
3. verified column sets, including the 2023 re-release's anonymized demographic columns
4. recorded the terms citation ("used solely for research, personal or non-commercial
   purposes") and the schema reference ("dunnhumby - The Complete Journey User Guide.pdf",
   inside the archive) in cj meta.json

The Kaggle mirror of this dataset is not used: dunnhumby grants use, not redistribution,
so the copy of record is dunnhumby's own asset.

## 6. What this cannot show, stated now

- Not Amex data, not card-network data: a grocery retailer's loyalty-card panel. The
  claim ceiling is "the protocol and the heads behave the same way on real
  transactional data", not "holds on real card data".
- The store outcome is sales growth over a fixed window, a coarse proxy for any business
  outcome a partner would care about.
- The offer head is predictive only (targeted campaigns).
- Small corpus: see section 2. If intervals are too wide to call, that is the result.
