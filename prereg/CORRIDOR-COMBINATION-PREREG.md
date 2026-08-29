# Pre-registration: corridor forecast combination (written before any computation)

Status: committed alone before the analysis runs. Nothing below changes after this commit:
no weight fitting, no alternative combinations tried, no threshold moved after seeing a
number. Whatever the computation shows ships, including a loss.

## Question

The corridor exhibit reports the model losing pooled accuracy to the per-corridor seasonal
naive (macro MASE 0.6230 vs 0.5302 over the 13-month holdout). The deployment question was
never whether the model replaces the naive but whether adding it helps. The standard answer
in forecasting is combination: Bates and Granger (1969) showed unweighted averages of
imperfectly correlated forecasts often beat both parents. This analysis tests exactly that,
once, with a fixed rule.

## Method, fixed now

- Forecasts: the shipped pipeline's one-month-ahead rolling holdout forecasts (2025-01 to
  2026-01, 13 months, 12 corridors) recomputed deterministically by the committed
  scripts/corridor_exhibit.py machinery from the committed SingStat input (the script is
  seeded and --check-able; the recomputation must reproduce the committed per-corridor MASE
  values at 1e-6 before the combination is evaluated, else the run aborts).
- Combination: the unweighted mean, 0.5 times naive plus 0.5 times model, per corridor per
  month. The 50/50 weight is fixed a priori (the equal-weights convention); no other weight
  is computed, reported, or tried.
- Metrics: macro mean per-corridor MASE of the combination (same MASE scale as shipped:
  in-sample mean absolute seasonal difference), reported against the shipped naive 0.5302
  and model 0.6230; per-corridor MASE table; count of corridors where the combination beats
  the naive.
- No interval is claimed for the pooled comparison (13 months, 12 corridors; the exhibit's
  own convention). The result is a point comparison and will be labelled as such.

## Decision rule, fixed now

- If combination macro MASE < naive macro MASE: the finding is that the model adds value as
  a complement to the naive, and it ships with the margin stated.
- If combination macro MASE >= naive macro MASE: the finding is that the model does not add
  value even as a complement at equal weights, and it ships with the same prominence.
- Either sentence is written by the producing script from the numbers, not by hand.

## Output

scripts/corridor_combination.py (deterministic, --check mode reproducing every numeric
leaf at 1e-6), results/corridor_combination.json (house shape: data sources, labels,
caveats, required_sentence). The page integration is ruled separately in
the project record before the run and happens whichever way the result lands.
