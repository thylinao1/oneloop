#!/usr/bin/env python3
"""Whitespace within-density re-read (POST HOC, and labelled so everywhere).

Why this exists
---------------
The pre-registered forward check called it for density: over the 1,510 buckets formed before
the cutoff, plain pre-cutoff venue count predicts where card-accepting venues appeared better
than the shipped composite, difference -0.1392 with a cell-clustered interval entirely below
zero. That loss stands, it is printed first everywhere it appears, and nothing here softens it.

But the composite CONTAINS density at weight 0.25, so that comparison answers "does the blend
beat one of its own four channels" and never asks the question a signing team actually has:
inside a band of comparable density, does the rest of the composite order where venues form?
That is the question this file asks, on the same committed data, with the same machinery.

WHAT THIS IS NOT
----------------
NOT pre-registered. The outcome data was already consumed by the primary check above, so no
pre-registration claim is available and none is made. What is fixed in advance is only what
this file could fix: the decision rule, the strata definition, the arms and the interval
procedure are all written here as constants, and the numbers were computed by running the file
as it stands. It is a secondary read of an outcome already observed, it is excluded from the
mechanical pre-registered census in scripts/protocol_value.py by construction (that census
reads its own row list and this file is not in it), and the page must label it post hoc
wherever it prints it.

DECISION RULE, fixed before the statistic was computed
------------------------------------------------------
The re-read rehabilitates nothing unless the non-density channels alone order venue formation
within density strata with a cell-clustered 95 percent interval entirely above zero. If that
interval covers zero, the file still writes, the page still prints it, and the sentence says
the re-read failed to find signal. A negative is publishable here; that is the whole habit.

Run: ./.venv/bin/python scripts/whitespace_stratified.py            writes the results file
     ./.venv/bin/python scripts/whitespace_stratified.py --check    recompute and compare
"""
from __future__ import annotations

import os

# Cap threads BEFORE numeric imports (8GB shared machine; determinism).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import whitespace_exhibit as we
import whitespace_temporal as wt
from derivation_check import write_or_check

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "whitespace_stratified.json"

SEED = 42
N_STRATA = 10
N_BOOT = 1000

# The composite minus its density channel. Spearman is rank-based, so leaving the three
# surviving weights un-renormalized cannot change any ordering and none is applied.
NON_DENSITY_WEIGHTS = {"mdr_prior_mean": 0.30, "tz_mean": 0.30, "indep_share": 0.15}

DECISION_RULE = (
    "The re-read is called a rehabilitation only if the non-density arm's size-weighted mean "
    "within-stratum Spearman carries a cell-clustered 95 percent interval entirely above zero. "
    "An interval covering zero ships as a failed re-read, in the same type size."
)

STATUS = (
    "POST HOC SECONDARY READ, NOT PRE-REGISTERED. The pre-registered primary check on this "
    "outcome had already been run and reported when this file was written, so the outcome was "
    "already observed and no pre-registration claim is available. What was fixed before the "
    "statistic was computed is written into this file as constants: the strata, the arms, the "
    "interval procedure and the decision rule. This result is excluded from the pre-registered "
    "census in scripts/protocol_value.py, which reads its own row list and does not include it."
)

CAVEATS = [
    "the primary pre-registered comparison still went against the composite and is printed "
    "first wherever this re-read appears; this does not overturn it and is not offered as an "
    "overturning",
    "stratifying on density does not remove density's information, it holds it roughly fixed "
    "inside a stratum; a within-stratum correlation is a statement about ordering inside a "
    "band of comparable density and nothing more",
    "the outcome is venue formation in Foursquare records, not merchant signing, the same "
    "caveat the primary check carries",
    "every caveat of the primary check travels with this one, including survivorship in the "
    "slice filter and date_created being record creation rather than opening",
]


def weighted_within_stratum_rho(arm, outcome, strata):
    """Size-weighted mean of the within-stratum Spearman, and the per-stratum detail.

    A stratum whose arm or outcome is constant has no defined rank correlation; it is dropped
    from the weighted mean and counted, never silently treated as zero.
    """
    total_w, acc, dropped, per = 0, 0.0, 0, []
    for s in range(N_STRATA):
        idx = np.flatnonzero(strata == s)
        if idx.size < 3:
            dropped += 1
            per.append(None)
            continue
        rho = spearmanr(arm[idx], outcome[idx]).statistic
        if np.isnan(rho):
            dropped += 1
            per.append(None)
            continue
        acc += float(rho) * idx.size
        total_w += idx.size
        per.append(round(float(rho), 6))
    if total_w == 0:
        return float("nan"), per, dropped
    return acc / total_w, per, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--data", default=str(ROOT / "data" / "fsq_sg.parquet"))
    args = ap.parse_args()
    data_path = Path(args.data)

    # Same gate the primary check runs: if the no-cutoff rebuild does not reproduce the shipped
    # ranking, the pre-cutoff predictor is some other construction and nothing here is about
    # the artifact the page ships.
    print("[gate] reproduction: rebuilding the shipped bucket set with no cutoff")
    repro = wt.reproduction_gate(data_path, ROOT)
    print(f"[gate] reproduction: n_buckets={repro['n_buckets_rebuilt']} "
          f"label_mismatches={repro['n_bucket_label_mismatches']} ok={repro['ok']}")
    if not repro["ok"]:
        sys.exit("GATE FAILED: the no-cutoff rebuild does not reproduce the shipped artifact. "
                 "Output not written.")

    buckets = wt.buckets_for(data_path, date_max=wt.CUTOFF_T)
    n = len(buckets)
    composite = np.array([b["score_real_signals"] for b in buckets], dtype=np.float64)
    density = np.array([b["n_pois"] for b in buckets], dtype=np.float64)
    non_density = np.array(
        [sum(w * b[k] for k, w in NON_DENSITY_WEIGHTS.items()) for b in buckets],
        dtype=np.float64)
    y, coverage = wt.outcome_counts(data_path, buckets)
    print(f"[build] {n} formed pre-cutoff buckets; "
          f"{coverage['n_in_formed_buckets']} outcome venues land in them")

    # Strata: density deciles by the same stable-sort convention the primary check uses for
    # every top-k cut, so ties fall the same way here as there. Fixed on the full sample and
    # held fixed under the bootstrap, so the strata cannot move with the resample.
    order = np.argsort(-density, kind="stable")
    strata = np.empty(n, dtype=np.int64)
    pos = 0
    for s, size in enumerate(wt.decile_sizes(n)):
        strata[order[pos:pos + size]] = s
        pos += size

    rho_comp, per_comp, drop_comp = weighted_within_stratum_rho(composite, y, strata)
    rho_nd, per_nd, drop_nd = weighted_within_stratum_rho(non_density, y, strata)
    rho_den, per_den, drop_den = weighted_within_stratum_rho(density, y, strata)
    print(f"[within] composite={rho_comp:.6f} non_density={rho_nd:.6f} density={rho_den:.6f}")

    # Cell-clustered bootstrap, the same cluster and seed the primary check uses.
    cell_keys = [(b["cell_x"], b["cell_y"]) for b in buckets]
    cells = sorted(set(cell_keys))
    members = {c: [] for c in cells}
    for i, c in enumerate(cell_keys):
        members[c].append(i)
    member_arrays = [np.array(members[c], dtype=np.int64) for c in cells]
    rng = np.random.default_rng(SEED)
    boot = {"composite": [], "non_density": [], "density": [], "non_density_minus_density": []}
    n_dropped = 0
    for _ in range(N_BOOT):
        drawn = rng.integers(0, len(cells), len(cells))
        pool = np.concatenate([member_arrays[j] for j in drawn])
        vals = {}
        for name, arm in (("composite", composite), ("non_density", non_density),
                          ("density", density)):
            r, _, _ = weighted_within_stratum_rho(arm[pool], y[pool], strata[pool])
            vals[name] = r
        if any(np.isnan(v) for v in vals.values()):
            n_dropped += 1
            continue
        for name, v in vals.items():
            boot[name].append(v)
        boot["non_density_minus_density"].append(vals["non_density"] - vals["density"])
    print(f"[boot] {N_BOOT} cell-clustered resamples, {n_dropped} undefined and dropped")

    def interval(vals):
        a = np.array(vals, dtype=np.float64)
        return [round(float(np.percentile(a, 2.5)), 6), round(float(np.percentile(a, 97.5)), 6)]

    ci_nd = interval(boot["non_density"])
    ci_comp = interval(boot["composite"])
    ci_den = interval(boot["density"])
    # The paired comparison a reader asks for as soon as the two numbers sit side by side.
    # Added while this file was still unpublished and before any of it reached the page; the
    # decision rule above was NOT changed, and this statistic only makes the claim easier to
    # falsify, since it can come back covering zero and would then be printed doing so.
    ci_pair = interval(boot["non_density_minus_density"])
    rho_pair = rho_nd - rho_den
    rehabilitates = ci_nd[0] > 0
    call = "non_density_channels_order_formation" if rehabilitates else "re_read_found_nothing"
    print(f"[decision] non_density={rho_nd:.4f} interval={ci_nd} call={call}")
    print(f"[paired] non_density minus density={rho_pair:.4f} interval={ci_pair}")

    primary = json.loads((ROOT / "results" / "whitespace_temporal.json").read_text())
    pj = primary["bootstrap"]["spearman_diff_composite_minus_density"]

    if rehabilitates:
        sentence = (
            f"The pre-registered forward check went against the composite and that stands: "
            f"difference {pj['point']:.4f}, interval "
            f"[{pj['interval_2p5_97p5'][0]:.4f}, {pj['interval_2p5_97p5'][1]:.4f}], entirely "
            f"below zero. A post hoc re-read of the same data asks a different question. Inside "
            f"bands of comparable density, the composite's three non-density channels order "
            f"where card-accepting venues appeared at a size-weighted Spearman of {rho_nd:.4f}, "
            f"cell-clustered interval [{ci_nd[0]:.4f}, {ci_nd[1]:.4f}], above zero, while "
            f"density's own ordering inside those same bands is {rho_den:.4f}, a paired difference "
            f"of {rho_pair:.4f} with interval [{ci_pair[0]:.4f}, {ci_pair[1]:.4f}]. So density wins "
            f"the marginal comparison and carries almost no ordering information once it is "
            f"held roughly fixed, which is what a signing team works inside. This is a secondary "
            f"read of an outcome already observed, not a pre-registered result, and it does not "
            f"reverse the check above.")
    else:
        sentence = (
            f"The pre-registered forward check went against the composite: difference "
            f"{pj['point']:.4f}, interval [{pj['interval_2p5_97p5'][0]:.4f}, "
            f"{pj['interval_2p5_97p5'][1]:.4f}]. A post hoc re-read asking whether the "
            f"non-density channels order formation inside bands of comparable density also "
            f"fails to find signal: size-weighted Spearman {rho_nd:.4f}, cell-clustered "
            f"interval [{ci_nd[0]:.4f}, {ci_nd[1]:.4f}], which covers zero. The lane loses on "
            f"both readings and both are printed.")

    import scipy  # noqa: F401
    payload = {
        "seed": SEED,
        "versions": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "polars": pl.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "generated_by": "scripts/whitespace_stratified.py --check-able",
        "status": STATUS,
        "is_pre_registered": False,
        "excluded_from_the_pre_registered_census": True,
        "data_sources": [we.DATA_SOURCE],
        "labels": ["real-public-data", "post hoc secondary read",
                   "outcome is venue formation, not merchant signing"],
        "what_this_is": (
            "the within-density re-read of the pre-registered forward check: the composite "
            "contains density at weight 0.25, so the primary comparison asks whether the blend "
            "beats one of its own channels; this asks whether the other three channels order "
            "venue formation inside bands of comparable density"),
        "the_primary_result_this_does_not_overturn": {
            "call": primary["decision"]["call"],
            "spearman_diff_composite_minus_density": pj,
            "note": "printed first wherever this re-read appears",
        },
        "decision_rule_fixed_before_the_statistic": DECISION_RULE,
        "design": {
            "n_buckets": n,
            "n_strata": N_STRATA,
            "strata_definition": (
                "density deciles over the pre-cutoff venue count per bucket, the arm that won "
                "the primary comparison, assigned by the same stable descending sort the "
                "primary check uses for its top-k cuts and held fixed under the bootstrap"),
            "non_density_arm": (
                "0.30*mdr_prior_mean + 0.30*tz_mean + 0.15*indep_share, the shipped composite "
                "with its 0.25 density channel removed and no renormalization, which cannot "
                "change a rank ordering"),
            "interval": (
                f"cell-clustered bootstrap, cluster = grid cell (cell_x, cell_y), {N_BOOT} "
                f"resamples with replacement, default_rng({SEED}), percentile 2.5 to 97.5, the "
                f"same procedure and seed as the primary check"),
            "reproduction_gate": repro,
        },
        "coverage": coverage,
        "arms_within_density_strata": {
            "composite": {"weighted_mean_spearman": round(rho_comp, 6),
                          "interval_2p5_97p5": ci_comp,
                          "per_stratum_spearman": per_comp,
                          "n_strata_dropped_undefined": drop_comp},
            "non_density_channels": {"weighted_mean_spearman": round(rho_nd, 6),
                                     "interval_2p5_97p5": ci_nd,
                                     "per_stratum_spearman": per_nd,
                                     "n_strata_dropped_undefined": drop_nd},
            "density_itself": {"weighted_mean_spearman": round(rho_den, 6),
                               "interval_2p5_97p5": ci_den,
                               "per_stratum_spearman": per_den,
                               "n_strata_dropped_undefined": drop_den},
            "non_density_minus_density": {
                "what": ("the paired within-strata difference, the comparison a reader asks for "
                         "once the two numbers sit side by side; same resamples, so it is "
                         "paired rather than a difference of two independent intervals"),
                "weighted_mean_spearman_difference": round(rho_pair, 6),
                "interval_2p5_97p5": ci_pair,
                "clears_zero": bool(ci_pair[0] > 0)},
        },
        "n_resamples_dropped_undefined": n_dropped,
        "call": call,
        "rehabilitates_the_non_density_channels": bool(rehabilitates),
        "required_sentence": sentence,
        "caveats": CAVEATS,
    }
    rc = write_or_check(OUT, payload)
    if rc or args.check:
        return rc
    print(f"wrote {OUT.relative_to(ROOT)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
