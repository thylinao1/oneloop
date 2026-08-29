# Pre-registration F-D: pair retention with the account-side as-of vector (finals menu, first item)

Status: written on 2026-08-25 under the approved finals menu (2026-08-24). It became a
pre-registration when committed to main unchanged. After that commit nothing below changes: no added arm, no swapped metric, no threshold
moved after seeing a number. Whatever the run shows ships, including a null, in finals material
only. The uploaded Round 1 artifact is never modified.

Trigger: the shortlist notification, or an organizer instruction to build a proof of concept. No
run before the trigger. If the finals brief conflicts with this item, the brief wins.

## 1. The question

The shipped page says, in Section 5 and again in Section 8, that the two-sided claim ("a backbone
that sees both the cardmember and the merchant beats one that sees a single side") is untested
rather than refuted, and it names the missing piece: the pair-retention arm carries the merchant's
pooled vector and nothing specific to the account, "and it is absent because this run was built to
stay on a laptop rather than because we decided it did not belong". This item runs that arm.

Plainly: pair retention asks whether an (account, merchant) pair that transacted before the corpus
cut transacts again after it. The shipped result (results/pair_retention.json) is a null for the
merchant-axis vector: delta PR-AUC +0.002829 with interval [-0.000422, 0.006311] over a strong
pair-history baseline. The account side is the half most likely to carry a return signal, and it
was never in the model.

## 2. What is held fixed, by reference to the shipped run

Everything in results/pair_retention.json `held_fixed`, `task` and `split` stays as shipped:

- Unit, label, cut (1503653820, 2017-08-25T09:37:00Z), corpus rows 24,386,900.
- Split: entity-disjoint by merchant, seeded hash partition, seed 7, test fraction 0.30, the same
  167,245 train merchants and 71,370 test merchants, 0 shared. The doubly disjoint split (merchant
  and account at once) is NOT the primary split here, because it would change the test rows and
  make the new arms incomparable with the shipped null; it is run as a secondary check (section 5).
- Baseline features: the 13 shipped pair, account and merchant pre-cut features.
- Model: LightGBM binary, the shipped parameter block, early stopping 50 rounds on a
  merchant-disjoint 15 percent validation slice of the train side, identical across arms.
- Bootstrap: merchant-clustered paired bootstrap, B = 1000, percentile 95 percent, seed 7.
- Every feature reads pre-cut rows only; every label reads post-cut rows only; the producer
  asserts `n_shared_merchants_train_test == 0` and `n_unmatched_merchant_rows_pre_cut == 0` before
  any number is written, as it does today.

## 3. The new input, defined before it is built

The account-side as-of vector: one 512-dimensional vector per account, produced by
scripts/fm/embed.py from the shipped cardholder-axis checkpoint (seed 7, the checkpoint the
transfer table and the ladder use, frozen, no retraining), pooled over that account's windows that
end strictly before the cut. No window that contains or follows a post-cut transaction contributes.
The join key is the hashed account id already used by the transfer evaluation. An account with no
pre-cut window cannot be a pair member by construction, so the join is total; the producer asserts
0 unmatched pairs and aborts otherwise.

## 4. Arms, and the one comparison that decides

Four arms, identical settings, only the feature block moves. Two are the shipped arms re-run
unchanged so the file reproduces its own history; two are new.

| Arm | Features | Status |
|---|---|---|
| A0 baseline | the 13 pair-history features | shipped, must reproduce at 1e-6 |
| A1 baseline plus merchant vector | A0 plus the 512-d merchant-axis pooled merchant embedding | shipped `with_v2`, must reproduce at 1e-6 |
| A2 baseline plus account vector | A0 plus the 512-d account-side as-of vector | new |
| A3 baseline plus both | A0 plus both vectors | new, the two-sided arm |

Primary comparison, fixed now: A3 minus A1 on PR-AUC. That is the question the page leaves open:
does the account side add measurable signal on top of the merchant side, on a task where the
baseline has real headroom (shipped baseline 0.863579 PR-AUC against a test positive rate of
0.177578).

Secondary comparisons, all reported, none deciding: A2 minus A0, A3 minus A0, A3 minus A2, on
PR-AUC and ROC-AUC, each with its paired interval.

Reproduction gate: A0 and A1 must reproduce the shipped `arms.baseline` and `arms.with_v2` values
at 1e-6 before any new arm is scored; otherwise the run aborts and nothing is reported.

## 5. Decision rule, frozen verbatim

The finals sentence is chosen by the producing script from the primary interval, not by hand:

- If the A3 minus A1 PR-AUC interval sits entirely above zero: "On the merchant-native task, adding
  the account-side vector to the merchant-side vector raised PR-AUC by [value], interval [lo, hi]
  above zero. That is the first two-sided gain on this page, measured on the synthetic corpus, and
  the size of it is [value] on a baseline of 0.863579."
- If the interval spans zero: "On the merchant-native task, adding the account-side vector to the
  merchant-side vector moved PR-AUC by [value], interval [lo, hi] spanning zero. The two-sided
  question stays open after the arm we said was missing, and it stays open on this page."
- If the interval sits entirely below zero: "On the merchant-native task, adding the account-side
  vector to the merchant-side vector lowered PR-AUC by [value], interval [lo, hi] below zero. On
  this corpus and this task the two-sided arm costs accuracy, and we print it."

Every sentence prints all four arms' values whenever any one is quoted. No sentence may quote A3
against A0 alone, because that comparison mixes the account-side question with the merchant-side
null already on the page.

### 5a. The split-naming rule (fixed on 2026-08-24, before any run)

The primary split is disjoint by merchant only. The corpus holds about 2,000 accounts across
838,863 pairs, so every account recurs across hundreds of pairs and sits on both sides of that
split. That is harmless for the shipped run, whose only added feature is merchant-side. It is not
harmless once an account-side vector enters: the model can learn account-specific behaviour from
an account's training pairs and apply it to the same account's test pairs, and a gain produced
that way is memorisation across the split rather than a two-sided mechanism. The
merchant-clustered bootstrap does not price it, because the correlation the account vector
introduces runs across merchants, the dimension the clustering treats as independent. So a
positive primary interval has two readings, and this file fixes now which one gets said.

The doubly disjoint split decides what the result is CALLED. Same four arms on a split disjoint
in merchants AND accounts at once (seeded hash partition on both, seed 7, keeping only pairs
whose account and merchant both fall on the same side; the producer records the pair counts it
keeps and drops). It scores different rows from the primary, so no paired interval is claimed
between the two splits; each split carries its own merchant-clustered paired interval on A3
minus A1.

| Primary (merchant-disjoint) A3 minus A1 | Doubly disjoint A3 minus A1 | What the result is called |
|---|---|---|
| interval above zero | interval above zero | a two-sided gain; both intervals are said |
| interval above zero | spans zero, or below zero | NOT ESTABLISHED; the sentence names account recurrence across the primary split as the reason the primary is not sufficient on its own; not a win and never presented as one |
| spans zero | any | the two-sided question stays open, the null sentence in section 5 applies, and the doubly disjoint result is reported alongside |
| below zero | any | the loss sentence in section 5 applies, and the doubly disjoint result is reported alongside |

The NOT ESTABLISHED sentence, drafted now: "On the merchant-native task the account-side vector
raised PR-AUC on the merchant-disjoint split by [value], interval [lo, hi], and did not on the
split that also holds out accounts, [value], interval [lo, hi]. Because the same accounts sit on
both sides of the first split, that gain is consistent with memorising an account across the
split, so it is not established as a two-sided mechanism, and we do not call it one."

The three outcome sentences in section 5 are read through this table: the "two-sided gain"
sentence may only be spoken in the first row.

## 6. Corpus and provenance

The corpus is the public IBM TabFormer benchmark, which is SYNTHETIC, and every finals sentence
carries that word. Nothing here is a measurement of American Express data. The result, whichever
way it lands, is a statement about a mechanism on a synthetic corpus.

The account universe is about 2,000 accounts. Written here before the run so it cannot become an
excuse after one: a null on this task may be a resolution limit of a 2,000-account corpus rather
than an absence of the mechanism, and the doubly disjoint split, which holds out accounts as
well, scores fewer pairs still. A null sentence may say "not resolved at this account count";
it may not say "the mechanism is absent", and a positive sentence gets no help from this
paragraph in either direction.

## 7. Output and cost

- Producer: scripts/pair_retention.py extended with `--stage run --account-vector <path>` or a
  sibling script that imports it; `--check` recomputes every numeric leaf at 1e-6; house
  envelope (seed, versions, generated_by, data_sources with sha256, labels including
  `synthetic`), arms and deltas as DICTS keyed by arm name, never lists, per the CONTRACT
  pointer law.
- Output: results/pair_retention_account_vector.json. The shipped results/pair_retention.json is
  never rewritten.
- Compute: the embedding pass over pre-cut windows for about 2,000 accounts is minutes on one
  GPU; the LightGBM arms are CPU, order of the shipped run. Under the two-job cluster budget, it
  fits in the first finals day. Exact accounting recorded from Slurm after the run.

## 8. Paired sentences for FINALS-PACK, drafted before the run

Win: "We said on the page which arm was missing. We ran it in the finals week under a rule we
wrote before we knew we would be here, and the account side added [value] PR-AUC on top of the
merchant side, interval above zero, on the synthetic corpus."

Null: "We said on the page which arm was missing. We ran it in the finals week under a rule we
wrote before we knew we would be here, and it came back a null. The two-sided claim is still a
hypothesis, and the pilot on real closed-loop data is where it gets settled."

Loss: "We said on the page which arm was missing. We ran it, and it cost accuracy. We would rather
tell you that than let the diagram imply otherwise."
