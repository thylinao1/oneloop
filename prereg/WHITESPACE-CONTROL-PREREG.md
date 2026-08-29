# WHITESPACE CONTROL, PRE-REGISTRATION

Written and committed BEFORE any control was computed. Nothing in this file was chosen after seeing a
result. If a later number disagrees with a rule written here, the rule stands and the deviation is
recorded in place, in the way `results/safety_membership.json#designed_not_run` records a
pre-registered exhibit that did not run.

Producer to be written: `scripts/safety/whitespace_control.py`
Output to be written: `results/whitespace_control.json`
Reference artifact under test: `results/whitespace.json`, frozen, produced by
`scripts/whitespace_exhibit.py`. That producer is NOT edited by this work and no leaf of
`results/whitespace.json` is moved by it.

---

## 0. Why this exists

The whitespace signing head is the only exhibit on the page that ships without a control. Every other
exhibit on the page carries one by policy, including one where the counts-only control beats the model
and ships anyway. The signing head reports a composite score per bucket and never states how its four
signals are combined, so a reader has no way to tell whether the composite is doing anything a single
column would not do.

This file registers the control. It does not register a validation, because no validation is available
here. See section 6.

---

## 1. The formula, recovered from the producer and stated before any control is run

Read out of `scripts/whitespace_exhibit.py` at the committed revision.

Per point of interest, at `compute_signals()` lines 361 to 364:

```
poi_score = 0.30 * mdr_sensitivity_prior
          + 0.30 * tourist_zone_proximity
          + 0.25 * local_density
          + 0.15 * independent_share
```

Per bucket, at `make_buckets()` line 396, the published `score_real_signals` is the unweighted mean of
`poi_score` over the points of interest in that bucket. The weighted sum is linear in the four
channels, so the bucket score is exactly the same weighted sum applied to the bucket's four channel
means:

```
score_real_signals(bucket) = 0.30 * mean(mdr_sensitivity_prior)
                           + 0.30 * mean(tourist_zone_score)
                           + 0.25 * mean(density_norm)
                           + 0.15 * mean(independent_share)
```

Those four bucket means are the four numbers already published per row at `ranking[i].signals`, so any
reader can recompute any released bucket by hand from the shipped file. The producer will write one
worked example into the output file so the recomputation is on the record and not merely asserted.

The four channels, each as the producer defines it:

- `mdr_sensitivity_prior`: a per category group constant from the `MDR_PRIORS` table, six distinct
  values, cited to public facts about Singapore card and QR pricing.
- `tourist_zone_proximity`: maximum over ten documented zones of `exp(-d_km / 1.5)` from the point to
  the zone centroid.
- `local_density`: `min(log1p(neighbours within 250 m) / log1p(p99), 1.0)`, neighbours counted by a
  KD-tree over the filtered universe, self excluded.
- `independent_share`: 1 when the normalized venue name appears fewer than three times in the
  Singapore slice, 0 otherwise.

**THE WEIGHTS ARE A HAND-SET JUDGEMENT CALL AND NOT A FITTED QUANTITY.** `WEIGHTS` at
`scripts/whitespace_exhibit.py:70` is a literal constant. Nothing in this repository fits it, tunes it,
cross-validates it or selects it against an outcome, because there is no outcome to select it against.
The four numbers were chosen by hand to sum to one. That sentence ships in the output file verbatim,
and the `equal_weights` arm registered below exists to measure how much the particular choice matters.

The simulated acceptance-gap signal is NOT in this formula. It appears only in the separate sensitivity
analysis at `sensitivity()` and never enters `score_real_signals` or the published ranking order.

---

## 2. Arms

One reference and seven control arms, plus three sanity rungs. Every arm scores the SAME bucket set,
rebuilt from the same frozen `data/fsq_sg.parquet` under its committed sha256, so no arm can win or
lose by scoring a different population.

**Reference**

- `composite`: the shipped `score_real_signals`. This is what the page ranks by today.

**Control arms**

- `raw_venue_count`: rank by the bucket's point-of-interest count `n_pois`, descending. Raw density,
  no normalization, no KD-tree, no weights. The dumbest thing that could work.
- `density_channel`: rank by `mean(density_norm)` alone, one of the composite's own four signals.
- `tourist_zone_channel`: rank by `mean(tourist_zone_score)` alone.
- `mdr_prior_channel`: rank by `mean(mdr_sensitivity_prior)` alone. Six distinct values across the
  whole bucket set, so it is heavily tied by construction and its numbers must be read that way.
- `independent_channel`: rank by `mean(independent_share)` alone.
- `equal_weights`: the same four channels at 0.25 each. This arm exists to price the hand-set weights.
- `embedding_wire`: rank by `score_with_embeddings`, the category-level backbone column already
  published in `results/whitespace.json`. Six distinct values, read from the committed file rather
  than recomputed, because recomputing it needs the embedding parquet and the TabFormer archive.

All four single-signal arms are registered together, and all four ship whatever they show. Registering
one and reporting one would let the choice of which single signal to compare against be made after
seeing which comparison flatters the composite.

**Sanity rungs, mandatory, because a pipeline that cannot report both poles reports nothing**

- `identity`: the composite scored against itself. MUST agree perfectly.
- `reversed`: the composite negated. MUST disagree totally.
- `random_uniform`: a seeded uniform random score per bucket, 20 seeds, mean and 2.5 to 97.5
  percentile spread across seeds in the shape `scripts/safety/privacy_ladder.py` uses. MUST be close
  to independent of the composite.

---

## 3. Metrics

Registered in the shape `results/safety_privacy_ladder.json` already uses for this same artifact, so
the two files read on the same axes.

- `spearman_full`: Spearman rank correlation between the composite score and the arm score over all
  buckets formed. Computed on the SCORES, so scipy assigns average ranks to ties. This matters for
  `mdr_prior_channel` and `embedding_wire`, which carry six values each.
- `spearman_released`: the same correlation restricted to the buckets the page actually releases, the
  composite's top 400 rows.
- `top20_overlap`, `top100_overlap`, `top400_overlap`: size of the set intersection between the
  composite's top k and the arm's top k. Ties inside a top-k cut are broken by the shipped stable
  order, score descending then the producer's own `(cell_x, cell_y, category_group)` sort, so the cut
  is deterministic and matches how the page itself resolves ties.
- `top20_left`, `top20_entered`, `top20_churn`: the churn triple, `churn = left + entered`.
  `top20_churn` is the headline, as it is in the privacy ladder.

---

## 4. The comparison, and the decision rule, both fixed now

The comparison is composite against each control arm. The question is not which ranking is better,
because nothing here can answer that. The question is how much of the composite's released list is
explained by one column.

Thresholds, fixed before the run:

- **NULL RESULT, and it ships as the headline if it lands.** If either density arm, `raw_venue_count`
  or `density_channel`, reaches `spearman_full >= 0.95` AND `top400_overlap >= 380` AND
  `top20_churn <= 4`, then the composite is an expensive way to sort by density, and the output file
  says so in those words. This outcome is worse for the submission and it ships without softening.
- **MATERIAL DIFFERENCE.** If the stronger of the two density arms shows `spearman_full <= 0.80` OR
  `top400_overlap <= 320`, the composite is doing something a density sort does not, and the file
  says that instead. It still does not say the composite is right, only that it is different.
- **BETWEEN THE TWO.** Report every number as obtained and attach no verdict word. An intermediate
  result is an intermediate result.

The same three thresholds are recorded per arm for the other five control arms, so a reader can see
which single column comes closest without a second decision rule being invented for it.

---

## 5. Sanity gates. Any failure voids the whole file

If any of these fails, `results/whitespace_control.json` is not written, the producer exits non-zero,
and the finding is that this producer cannot be trusted rather than anything about the composite.

- `identity`: `spearman_full == 1.0` and `spearman_released == 1.0` exactly; overlaps 20, 100, 400;
  `top20_churn == 0`.
- `reversed`: `spearman_full == -1.0` exactly; `top20_overlap == 0`; `top400_overlap == 0`, which is
  forced because 400 plus 400 is below the bucket count.
- `random_uniform`: mean `|spearman_full| < 0.10` across the 20 seeds, every individual seed
  `|spearman_full| < 0.15`, and mean `top20_overlap < 2`.
- **Reproduction of the shipped artifact.** The rebuilt bucket set must reproduce the committed
  released list: 400 rows, bucket labels identical in order, `score_real_signals` within 1e-6. Without
  this the control is measuring some other ranking.
- **Cross-check against a published number.** The `embedding_wire` arm must reproduce the two
  rank correlations already published at `results/whitespace.json#wire.rank_reorder`, 0.0451 over the
  released 400 and -0.1276 over all buckets, within 1e-4. They are stored rounded to four decimals,
  which is why the tolerance is 1e-4 there and 1e-6 everywhere else.

---

## 6. What this does NOT do, stated before the run so it cannot be quietly dropped after it

There is no observed merchant-acceptance label anywhere in this corpus. Foursquare venue records say
what a venue is and where it is. They do not say whether that venue was ever approached, ever signed,
or ever would sign.

So:

- No arm can be shown to predict real signings. This work adds a control and a stated formula. It does
  not add validation, a holdout or a ground truth, and no wording in the output file or on the page may
  imply that it does.
- No proxy outcome will be constructed. Building a label out of the same signals that produced the
  ranking and then reporting agreement with it would be a validation-shaped object with no validation
  in it, which is worse than having no control at all.
- The acceptance-gap signal stays simulated, and stays labelled `simulated-increment`. That limit is
  already disclosed on the page and stays disclosed.
- The word baseline in this file means a simpler ranking to compare against. It never means a model to
  beat.

## 7. What would make the result meaningless

Recorded now so it cannot be rationalized later.

- Any sanity gate failing. Covered in section 5.
- Reading a rank agreement as evidence of accuracy. Two rankings agreeing tells you they order the
  same buckets the same way. Neither of them is thereby correct.
- Quoting `mdr_prior_channel` or `embedding_wire` correlations without saying that both carry six
  distinct values across more than a thousand buckets, so their ranks are mostly ties.
- Treating `equal_weights` agreement as a defence of the specific weights. High agreement there means
  the exact weights barely matter, which is an argument that the weighting is not load bearing, not an
  argument that it is right.

---

Committed before the producer was written. Seed 42, laptop, CPU, deterministic, `--check` at 1e-6 in
the shape `scripts/safety/privacy_ladder.py --check` uses.
