# WHITESPACE TEMPORAL HOLDOUT, PRE-REGISTRATION

Written and committed BEFORE any score-versus-outcome number was computed. The only data looked at
before writing this file: the `date_created` histogram of `data/fsq_sg.parquet` (yearly over the full
slice, monthly from 2024 within the card-accepting category universe) and the pre and post venue
counts at four candidate cutoffs. Those are feasibility quantities for the cutoff rule in section 2.
No predictor was scored against any outcome before this file was committed. If a later number
disagrees with a rule written here, the rule stands and the deviation is recorded in place.

Producer to be written: `scripts/whitespace_temporal.py`
Output to be written: `results/whitespace_temporal.json`
Reference artifact: `results/whitespace.json` (frozen) produced by `scripts/whitespace_exhibit.py`.
Neither file is modified by this work. No shipped file is modified by this work. The only commit this
workstream makes is the commit of this pre-registration.

---

## 0. What this is and what it is not

The shipped whitespace exhibit ranks 1,536 bucket units (a ~0.008 degree grid cell crossed with a
category group) by a composite of four real signals. It ships with a control
(`results/whitespace_control.json`) but with no outcome: public data does not record which merchants
signed for card acceptance, and WHITESPACE-CONTROL-PREREG.md section 6 says so.

This work adds the one forward-looking check the data supports: the slice kept `date_created`, so the
composite computed from venues that existed before a cutoff can be tested against where new venues in
the card-accepting universe actually appeared after the cutoff.

What this is NOT, fixed now:

- The outcome is VENUE FORMATION, not merchant signing. A place opening is not a merchant signing a
  card-acceptance agreement. No wording in the output may equate the two.
- `date_created` is the date the record entered Foursquare's system, a proxy for the venue's opening
  date. Record creation can lag opening or batch-arrive in ingest waves.
- SURVIVORSHIP: the slice filter was `country='SG' AND date_closed IS NULL`, so venues created before
  the cutoff that closed before the 2026-08-11 snapshot are absent. Pre-cutoff predictors are computed
  on the survivors, not on the true pre-cutoff population.
- Beating density here does not make the composite a signing model. Losing to density ships as the
  finding all the same.

---

## 1. What the histogram showed (recorded because the cutoff rule reads it)

Full slice (364,635 rows, `date_created` spans 2003-12-10 to 2026-08-10): large ingest waves in
2010 (83,513), 2011 (119,938), 2012 (42,413), then a declining tail, then a steady stream of roughly
4,000 to 6,000 per year from 2020 on.

Card-accepting category universe (the exhibit's own filter and group rules, 146,048 venues): monthly
creations from 2024-01 onward run steadily between 318 and 468 per month with no gaps. Candidate
cutoffs measured for feasibility only:

| cutoff T | pre-T universe venues | post-T universe venues |
|---|---|---|
| 2023-08-10 | 131,891 | 14,157 |
| 2024-01-01 | 133,661 | 12,387 |
| 2024-08-10 | 136,418 | 9,630 |
| 2025-01-01 | 138,357 | 7,691 |

---

## 2. Cutoff rule, fixed now

Snapshot S = 2026-08-10, the latest `date_created` observed in the slice (release dt=2026-08-11).

Rule: take the LONGEST outcome window between 12 and 24 months ending at S whose outcome set (venues
in the card-accepting category universe with `date_created` in (T, S]) holds at least 5,000 venues.
The 24-month window qualifies (9,630 >= 5,000), so:

- **T = 2024-08-10.** Predictor venues: `date_created` <= T. Outcome venues: `date_created` in
  (T, S]. Comparison on ISO date strings, which is safe for this format.

---

## 3. Unit, predictors, outcome

**Unit.** The exhibit's own bucket: `(cell_x, cell_y, category_group)` with `cell = floor(coord /
0.008 deg)`, formed when it holds at least 10 predictor venues (MIN_BUCKET_POIS, applied to the pre-T
subset). The analysis population is all buckets formed pre-T. Its size will be near but not equal to
the shipped 1,536 and is reported as obtained.

**Predictors, each computed ONLY from venues with `date_created` <= T.** The producer reuses the
exhibit's own functions (`assign_group`, `compute_signals`, `make_buckets` imported from
`scripts/whitespace_exhibit.py`, which is not edited); only the frame construction is refactored to
accept the date predicate. Everything the construction estimates from data is re-estimated within the
pre-T subset: the chain rule (name frequency >= 3), the density KD-tree neighbor counts, and the p99
density normalizer. The tourist zones, the MDR priors, and the weights 0.30/0.30/0.25/0.15 are
constants and stay untouched.

- `composite`: the shipped `score_real_signals` construction on pre-T venues. The arm under test.
- `raw_venue_count`: pre-T venue count per bucket, descending. THE NULL HYPOTHESIS. If new venues
  simply appear where venues already are, this wins and that ships as the finding.
- `equal_weights`: the same four channel means at 0.25 each.
- `tourist_zone_channel`: the pre-T mean tourist-zone score alone.
- `random`: 1,000 seeded random orderings of the same buckets (numpy default_rng(42)); mean and
  2.5 to 97.5 percentile spread across draws; analytic expectations stated beside them.

**Outcome.** Per formed bucket: the count of card-accepting-universe venues with `date_created` in
(T, S] falling in that same `(cell_x, cell_y, category_group)`. Zero when none. The group assignment
for outcome venues uses the same `assign_group` rules. Post-T venues landing where no pre-T bucket
was formed are outside the analysis population; their count and share are reported as a descriptive
coverage statistic, not as a metric.

**The backbone wire, documented now.** `score_with_embeddings` is a category-level column (six
distinct values, one per category group, from TabFormer backbone embeddings via
`data/merchant_embeddings.parquet`). It enters neither `score_real_signals` nor the shipped ranking
order. Because it is constant per category group, restricting venues to pre-T changes which buckets
exist but cannot change any bucket's wire value. It is therefore not recomputed and not an arm here;
this validation tests the composite the page actually ranks by.

---

## 4. Metrics, fixed now

Ties inside any top-k cut are broken by the stable order `(cell_x, cell_y, category_group)`
ascending, the shipped producer's own sort. Spearman is scipy `spearmanr` on scores (average ranks
for ties).

- `spearman_vs_outcome`: Spearman between the arm's score and the outcome count over all formed
  buckets. Reported for every arm.
- `precision_at_50`: the size of the intersection between the arm's top 50 buckets and the outcome's
  top 50 buckets, divided by 50. The outcome's top-50 cut uses the stable tie-break; the outcome
  value at the boundary and the number of buckets tied at it are reported beside the metric because
  the tie-break inside that boundary is arbitrary.
- `lift_by_decile`: buckets sorted by the arm's score descending (stable tie-break), split into 10
  contiguous deciles as equal as possible (the first `N mod 10` deciles get the extra bucket); lift
  of a decile = mean outcome in the decile / mean outcome over all formed buckets. The headline is
  the top decile's lift.

**Uncertainty, primary comparison.** Cell-clustered bootstrap: the cluster is the grid cell
`(cell_x, cell_y)`; draw n_cells cells with replacement (numpy default_rng(42), 1,000 resamples);
each resample pools every formed bucket of every drawn cell, with multiplicity; on the pooled sample
compute Spearman(composite, outcome) minus Spearman(raw_venue_count, outcome). Report the point
difference and the 2.5 to 97.5 percentile interval. Resamples where either Spearman is undefined
(constant vector) are dropped and counted; the count ships in the output.

The same bootstrap machinery also reports the interval for the composite's own
`spearman_vs_outcome`. No other bootstrap comparisons are registered and none will be added.

---

## 5. Decision rule, fixed now

The primary question: does the composite predict where card-accepting-universe venues appear next
better than plain pre-T venue count. Thresholds:

- **COMPOSITE BEATS DENSITY** if the bootstrap 95 percent interval (2.5 to 97.5 percentile) of the
  Spearman difference (composite minus raw_venue_count) lies entirely above zero.
- **DENSITY BEATS COMPOSITE, and it ships as the headline in those words** if that interval lies
  entirely below zero.
- **NOT RESOLVED** if the interval covers zero. The honest reading is then that the composite is not
  shown to add predictive value over density on this outcome, and the output says that.

`equal_weights` and `tourist_zone_channel` results are reported as obtained with no verdict word
attached; they exist so a reader can see whether the specific weights or one channel carry the
result. The random arm exists to anchor chance level.

---

## 6. Sanity gates. Any failure voids the file

If any gate fails, `results/whitespace_temporal.json` is not written, the producer exits non-zero,
and the finding is that the producer cannot be trusted, not anything about the composite.

- **Reproduction of the shipped artifact.** Run the same reused construction with NO date cutoff:
  it must rebuild the shipped bucket set (1,536 buckets) and reproduce the committed released list in
  `results/whitespace.json`: 400 rows, bucket labels identical in order, `score_real_signals` within
  1e-6. Without this the pre-T predictor is some other construction.
- **Identity.** Spearman(outcome, outcome) = 1 within 1e-6.
- **Random anchor.** Across the 1,000 random draws: mean |spearman_vs_outcome| < 0.05 and every
  individual draw |spearman_vs_outcome| < 0.15.

---

## 7. Output conventions

- `scripts/whitespace_temporal.py`: deterministic, seed 42, thread caps as the exhibit sets them,
  runnable end to end on the laptop, `--check` recomputes and compares every numeric leaf of the
  committed JSON at 1e-6 in the house pattern.
- `results/whitespace_temporal.json` restates the thresholds of section 5, carries every metric with
  its interval, all arms, the coverage statistic, the sanity gate results, a `caveats` list that
  includes at least: (i) `date_created` is Foursquare record creation, a proxy for venue opening;
  (ii) survivorship, as stated in section 0; (iii) the outcome is venue formation, not merchant
  signing; (iv) the wire documentation of section 3. It carries a `required_sentence` field whose
  text branches on the section 5 outcome and states the honest reading, including a null or a loss
  to density.
- Tone: no em dashes, no en dashes, plain professional language.

Committed alone, before the producer was written and before any outcome was computed. Laptop, CPU,
deterministic throughout.
