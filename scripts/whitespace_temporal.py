#!/usr/bin/env python3
"""Whitespace temporal holdout (W-TEMPORAL, fix L-C).

Pre-registered in WHITESPACE-TEMPORAL-PREREG.md, committed before this producer
was written and before any score-versus-outcome number was computed.

Design: score the exhibit's own buckets using ONLY venues with date_created <=
T = 2024-08-10, then test each ranking against where card-accepting-universe
venues actually appeared in (T, 2026-08-10]. Arms: the shipped composite, raw
pre-T venue count (the null), equal weights, the tourist-zone channel alone,
and 1,000 seeded random orderings. Primary decision: the cell-clustered
bootstrap 95 percent interval of Spearman(composite, outcome) minus
Spearman(raw_venue_count, outcome).

The exhibit's construction is reused, not reimplemented: assign_group,
compute_signals and make_buckets are imported from scripts/whitespace_exhibit.py
(not edited); only the frame construction is refactored here to accept the date
predicate. A no-cutoff rebuild must reproduce the shipped released list in
results/whitespace.json (sanity gate; any gate failure voids the output).

--check: recompute deterministically and compare every numeric leaf of the
committed results/whitespace_temporal.json at 1e-6 (house pattern).
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

SEED = 42
CUTOFF_T = "2024-08-10"       # prereg section 2: 24-month window ending at snapshot
SNAPSHOT_S = "2026-08-10"     # latest date_created in the slice (release dt=2026-08-11)
N_BOOT = 1000
N_RANDOM = 1000
PRECISION_K = 50
N_DECILES = 10
TOL = 1e-6
RANDOM_MEAN_ABS_RHO_MAX = 0.05
RANDOM_PER_DRAW_ABS_RHO_MAX = 0.15

PREREG = {
    "file": "WHITESPACE-TEMPORAL-PREREG.md",
    "committed_before_any_outcome_was_computed": True,
    "cutoff_rule": ("longest outcome window between 12 and 24 months ending at the snapshot "
                    "whose card-accepting-universe outcome set holds at least 5,000 venues; "
                    "the 24-month window qualified (9,630 >= 5,000), so T = 2024-08-10"),
    "decision_rule": {
        "composite_beats_density": ("bootstrap 95 percent interval (2.5 to 97.5 percentile) of "
                                    "Spearman(composite, outcome) minus "
                                    "Spearman(raw_venue_count, outcome) lies entirely above zero"),
        "density_beats_composite": "that interval lies entirely below zero",
        "not_resolved": "the interval covers zero",
    },
    "sanity_gates": {
        "reproduction": ("no-cutoff rebuild reproduces results/whitespace.json: 1,536 buckets, "
                         "400 released rows, labels identical in order, scores within 1e-6"),
        "identity": "Spearman(outcome, outcome) = 1 within 1e-6",
        "random_anchor": (f"mean |spearman| < {RANDOM_MEAN_ABS_RHO_MAX} and every draw "
                          f"|spearman| < {RANDOM_PER_DRAW_ABS_RHO_MAX} across the "
                          f"{N_RANDOM} random draws"),
    },
}

CAVEATS = [
    "date_created is the date the record entered Foursquare's system, a proxy for the venue's "
    "opening date; record creation can lag opening or batch-arrive in ingest waves",
    "survivorship: the slice filter was country='SG' AND date_closed IS NULL, so venues created "
    "before the cutoff that closed before the 2026-08-11 snapshot are absent; pre-cutoff "
    "predictors are computed on the survivors, not on the true pre-cutoff population",
    "the outcome is venue formation, not merchant signing; a place opening is not a merchant "
    "signing a card-acceptance agreement, and no arm here is validated against signings",
    "the backbone wire (score_with_embeddings) is a category-level column, constant per category "
    "group; it enters neither score_real_signals nor the shipped ranking order, and restricting "
    "venues to pre-T changes which buckets exist but cannot change any bucket's wire value, so it "
    "was not recomputed and is not an arm here; this validation tests the composite the page "
    "actually ranks by",
    "post-cutoff venues landing where no pre-cutoff bucket was formed are outside the analysis "
    "population; their count is reported under coverage and is not part of any metric",
]


# ---------------------------------------------------------------- construction --

def build_frame(data_path: Path, date_max: str | None) -> pl.DataFrame:
    """The exhibit's build_poi_frame with an optional date_created predicate.

    Everything estimated from data (group assignment, chain rule) is estimated
    within the returned subset, so a pre-T frame uses only pre-T information.
    """
    df = pl.read_parquet(data_path)
    cond = (
        pl.col("date_closed").is_null()
        & pl.col("latitude").is_between(1.15, 1.48)
        & pl.col("longitude").is_between(103.6, 104.1)
        & pl.col("fsq_category_labels").is_not_null()
        & pl.col("name").is_not_null()
    )
    if date_max is not None:
        cond = cond & (pl.col("date_created") <= date_max)
    df = df.filter(cond)
    groups = [we.assign_group(labels) for labels in df["fsq_category_labels"].to_list()]
    df = df.with_columns(pl.Series("category_group", groups, dtype=pl.Utf8))
    df = df.filter(pl.col("category_group").is_not_null())
    df = df.with_columns(
        pl.col("name").str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()
        .alias("norm_name")
    )
    counts = df.group_by("norm_name").len().rename({"len": "name_count"})
    df = df.join(counts, on="norm_name", how="left")
    df = df.with_columns((pl.col("name_count") >= we.CHAIN_MIN_COUNT).alias("is_chain"))
    return df


def buckets_for(data_path: Path, date_max: str | None) -> list[dict]:
    df = build_frame(data_path, date_max)
    df = we.compute_signals(df)
    return we.make_buckets(df)


def outcome_counts(data_path: Path, buckets: list[dict]) -> tuple[np.ndarray, dict]:
    """Outcome per formed bucket: card-accepting-universe venues created in (T, S]."""
    df = pl.read_parquet(data_path)
    df = df.filter(
        pl.col("date_closed").is_null()
        & pl.col("latitude").is_between(1.15, 1.48)
        & pl.col("longitude").is_between(103.6, 104.1)
        & pl.col("fsq_category_labels").is_not_null()
        & pl.col("name").is_not_null()
        & (pl.col("date_created") > CUTOFF_T)
        & (pl.col("date_created") <= SNAPSHOT_S)
    )
    groups = [we.assign_group(labels) for labels in df["fsq_category_labels"].to_list()]
    df = df.with_columns(pl.Series("category_group", groups, dtype=pl.Utf8))
    df = df.filter(pl.col("category_group").is_not_null())
    df = df.with_columns(
        (pl.col("latitude") // we.GRID_DEG).cast(pl.Int32).alias("cell_y"),
        (pl.col("longitude") // we.GRID_DEG).cast(pl.Int32).alias("cell_x"),
    )
    n_total = df.height
    agg = df.group_by(["cell_x", "cell_y", "category_group"]).len()
    count_by_key = {(r["cell_x"], r["cell_y"], r["category_group"]): r["len"]
                    for r in agg.to_dicts()}
    y = np.zeros(len(buckets), dtype=np.int64)
    formed_keys = set()
    for i, b in enumerate(buckets):
        key = (b["cell_x"], b["cell_y"], b["category_group"])
        formed_keys.add(key)
        y[i] = count_by_key.get(key, 0)
    n_in_formed = int(y.sum())
    coverage = {
        "n_outcome_venues_total": int(n_total),
        "n_in_formed_buckets": n_in_formed,
        "n_outside_formed_buckets": int(n_total - n_in_formed),
        "share_in_formed_buckets": round(n_in_formed / n_total, 4) if n_total else None,
        "n_outcome_bucket_keys_not_formed_pre_cutoff": int(
            sum(1 for k in count_by_key if k not in formed_keys)),
        "note": ("venues outside formed buckets fall where fewer than 10 pre-cutoff venues "
                 "existed; they are a descriptive statistic, not part of any metric"),
    }
    return y, coverage


# --------------------------------------------------------------------- metrics --

def stable_top(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top k by score descending; ties keep the stable
    (cell_x, cell_y, category_group) order, which is the bucket list order."""
    return np.argsort(-scores.astype(np.float64), kind="stable")[:k]


def precision_at_k(arm: np.ndarray, outcome: np.ndarray, k: int) -> dict:
    top_arm = set(stable_top(arm, k).tolist())
    top_out_idx = stable_top(outcome.astype(np.float64), k)
    boundary = int(outcome[top_out_idx[-1]])
    return {
        "precision_at_50": round(len(top_arm & set(top_out_idx.tolist())) / k, 4),
        "outcome_at_boundary": boundary,
        "n_buckets_tied_at_boundary_total": int((outcome == boundary).sum()),
        "n_tied_at_boundary_inside_cut": int((outcome[top_out_idx] == boundary).sum()),
    }


def decile_sizes(n: int) -> list[int]:
    base, extra = divmod(n, N_DECILES)
    return [base + 1 if d < extra else base for d in range(N_DECILES)]


def lift_by_decile(arm: np.ndarray, outcome: np.ndarray) -> list[float]:
    order = np.argsort(-arm.astype(np.float64), kind="stable")
    mean_all = float(outcome.mean())
    lifts, pos = [], 0
    for size in decile_sizes(len(arm)):
        part = order[pos:pos + size]
        lifts.append(round(float(outcome[part].mean()) / mean_all, 4))
        pos += size
    return lifts


def arm_metrics(arm: np.ndarray, outcome: np.ndarray) -> dict:
    rho = float(spearmanr(arm, outcome).statistic)
    lifts = lift_by_decile(arm, outcome)
    return {
        "spearman_vs_outcome": round(rho, 6),
        **precision_at_k(arm, outcome, PRECISION_K),
        "lift_top_decile": lifts[0],
        "lift_by_decile": lifts,
    }


def pct_interval(vals: np.ndarray) -> list[float]:
    return [round(float(np.percentile(vals, 2.5)), 6),
            round(float(np.percentile(vals, 97.5)), 6)]


def random_arm(outcome: np.ndarray) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(SEED)
    n = len(outcome)
    rhos = np.empty(N_RANDOM)
    precs = np.empty(N_RANDOM)
    lift_top = np.empty(N_RANDOM)
    for d in range(N_RANDOM):
        scores = rng.random(n)
        rhos[d] = spearmanr(scores, outcome).statistic
        precs[d] = precision_at_k(scores, outcome, PRECISION_K)["precision_at_50"]
        lift_top[d] = lift_by_decile(scores, outcome)[0]
    out = {
        "what": f"{N_RANDOM} seeded random orderings of the same buckets (default_rng({SEED}))",
        "spearman_vs_outcome": {"mean": round(float(rhos.mean()), 6),
                                "interval_2p5_97p5": pct_interval(rhos),
                                "max_abs": round(float(np.abs(rhos).max()), 6),
                                "analytic_expectation": 0.0},
        "precision_at_50": {"mean": round(float(precs.mean()), 4),
                            "interval_2p5_97p5": pct_interval(precs),
                            "analytic_expectation": round(PRECISION_K / n, 4)},
        "lift_top_decile": {"mean": round(float(lift_top.mean()), 4),
                            "interval_2p5_97p5": pct_interval(lift_top),
                            "analytic_expectation": 1.0},
        "n_draws": N_RANDOM,
    }
    return out, rhos


def cluster_bootstrap(buckets: list[dict], composite: np.ndarray,
                      density: np.ndarray, outcome: np.ndarray) -> dict:
    """Cell-clustered bootstrap of the Spearman difference composite minus density.

    Cluster = grid cell (cell_x, cell_y); every formed bucket of a drawn cell
    enters the resample, with multiplicity.
    """
    cell_keys = [(b["cell_x"], b["cell_y"]) for b in buckets]
    cells = sorted(set(cell_keys))
    members: dict[tuple, list[int]] = {c: [] for c in cells}
    for i, c in enumerate(cell_keys):
        members[c].append(i)
    member_arrays = [np.array(members[c], dtype=np.int64) for c in cells]
    n_cells = len(cells)

    rng = np.random.default_rng(SEED)
    diffs, comps = [], []
    n_dropped = 0
    for _ in range(N_BOOT):
        drawn = rng.integers(0, n_cells, n_cells)
        pool = np.concatenate([member_arrays[j] for j in drawn])
        rc = spearmanr(composite[pool], outcome[pool]).statistic
        rd = spearmanr(density[pool], outcome[pool]).statistic
        if np.isnan(rc) or np.isnan(rd):
            n_dropped += 1
            continue
        diffs.append(rc - rd)
        comps.append(rc)
    diffs, comps = np.array(diffs), np.array(comps)
    point = (float(spearmanr(composite, outcome).statistic)
             - float(spearmanr(density, outcome).statistic))
    return {
        "what": ("cell-clustered bootstrap: cluster = grid cell (cell_x, cell_y), "
                 f"{N_BOOT} resamples of {n_cells} cells with replacement, "
                 f"default_rng({SEED}), percentile 2.5 to 97.5"),
        "n_cells": n_cells,
        "n_resamples": N_BOOT,
        "n_dropped_undefined": n_dropped,
        "spearman_diff_composite_minus_density": {
            "point": round(point, 6),
            "interval_2p5_97p5": pct_interval(diffs),
        },
        "spearman_composite": {
            "point": round(float(spearmanr(composite, outcome).statistic), 6),
            "interval_2p5_97p5": pct_interval(comps),
        },
    }


# ------------------------------------------------------------ gates and verdict --

def reproduction_gate(data_path: Path, repo: Path) -> dict:
    shipped = json.loads((repo / "results" / "whitespace.json").read_text())
    buckets = buckets_for(data_path, date_max=None)
    scores = np.array([b["score_real_signals"] for b in buckets], dtype=np.float64)
    order = np.argsort(-scores, kind="stable")[:len(shipped["ranking"])]
    label_mismatches = 0
    max_diff = 0.0
    for pos, row in zip(order, shipped["ranking"]):
        b = buckets[int(pos)]
        if b["bucket_label"] != row["bucket_label"]:
            label_mismatches += 1
        max_diff = max(max_diff, abs(float(b["score_real_signals"])
                                     - row["score_real_signals"]))
    gate = {
        "n_buckets_rebuilt": len(buckets),
        "n_buckets_shipped": shipped["n_buckets"],
        "n_released_rows_compared": len(shipped["ranking"]),
        "n_bucket_label_mismatches": label_mismatches,
        "max_abs_score_diff": max_diff,
        "tolerance": TOL,
        "ok": (len(buckets) == shipped["n_buckets"] and label_mismatches == 0
               and max_diff <= TOL),
    }
    return gate


def _required_sentence(call: str, s: dict) -> str:
    """The sentence the record must carry, branched on the pre-registered outcome."""
    base = (f"Over the {s['n_buckets']} buckets formed before {CUTOFF_T}, ")
    tail = (" The outcome is venue formation in Foursquare records, not merchant signing, and "
            "date_created is record creation, a proxy for opening, so this is a forward-looking "
            "check of the ranking against real subsequent data, not a validation against signings.")
    if call == "composite_beats_density":
        return (base + f"the shipped composite predicts where card-accepting venues appeared in "
                f"the following 24 months better than plain venue count: Spearman "
                f"{s['rho_composite']:.4f} against {s['rho_density']:.4f}, difference "
                f"{s['diff_point']:.4f} with a cell-clustered bootstrap 95 percent interval of "
                f"[{s['diff_lo']:.4f}, {s['diff_hi']:.4f}], clear of zero." + tail)
    if call == "density_beats_composite":
        return (base + f"plain pre-cutoff venue count predicts where card-accepting venues "
                f"appeared in the following 24 months better than the shipped composite: Spearman "
                f"{s['rho_density']:.4f} against the composite's {s['rho_composite']:.4f}, "
                f"difference {s['diff_point']:.4f} with a cell-clustered bootstrap 95 percent "
                f"interval of [{s['diff_lo']:.4f}, {s['diff_hi']:.4f}], entirely below zero. "
                f"The composite is not shown to add predictive value over density on this "
                f"outcome, and that is the finding." + tail)
    return (base + f"the composite and plain pre-cutoff venue count are not separated on this "
            f"outcome: Spearman {s['rho_composite']:.4f} against {s['rho_density']:.4f}, "
            f"difference {s['diff_point']:.4f} with a cell-clustered bootstrap 95 percent "
            f"interval of [{s['diff_lo']:.4f}, {s['diff_hi']:.4f}], which covers zero. The "
            f"composite is not shown to add predictive value over plain density on this "
            f"outcome." + tail)


# ------------------------------------------------------------------------- run --

def build(data_path: Path, repo: Path) -> dict:
    print(f"[gate] reproduction: rebuilding the shipped bucket set with no cutoff")
    repro = reproduction_gate(data_path, repo)
    print(f"[gate] reproduction: n_buckets={repro['n_buckets_rebuilt']} "
          f"label_mismatches={repro['n_bucket_label_mismatches']} "
          f"max_abs_score_diff={repro['max_abs_score_diff']:.2e} ok={repro['ok']}")
    if not repro["ok"]:
        sys.exit("GATE FAILED: the no-cutoff rebuild does not reproduce the shipped artifact; "
                 "the pre-T predictor would be some other construction. Output not written.")

    print(f"[build] pre-cutoff buckets (date_created <= {CUTOFF_T})")
    buckets = buckets_for(data_path, date_max=CUTOFF_T)
    n = len(buckets)
    composite = np.array([b["score_real_signals"] for b in buckets], dtype=np.float64)
    raw_count = np.array([b["n_pois"] for b in buckets], dtype=np.float64)
    tz = np.array([b["tz_mean"] for b in buckets], dtype=np.float64)
    equal = np.array([0.25 * (b["mdr_prior_mean"] + b["tz_mean"] + b["dens_mean"]
                              + b["indep_share"]) for b in buckets], dtype=np.float64)
    print(f"[build] {n} formed pre-cutoff buckets")

    y, coverage = outcome_counts(data_path, buckets)
    print(f"[build] outcome: {coverage['n_outcome_venues_total']} venues in "
          f"({CUTOFF_T}, {SNAPSHOT_S}]; {coverage['n_in_formed_buckets']} land in formed buckets")
    if y.mean() == 0:
        sys.exit("GATE FAILED: outcome is all zero over formed buckets; lift is undefined. "
                 "Output not written.")

    identity_rho = float(spearmanr(y, y).statistic)
    identity_ok = abs(identity_rho - 1.0) <= TOL
    print(f"[gate] identity: spearman(outcome, outcome)={identity_rho} ok={identity_ok}")

    print(f"[arms] scoring {N_RANDOM} random draws")
    rand_out, rand_rhos = random_arm(y)
    random_ok = (float(np.abs(rand_rhos).mean()) < RANDOM_MEAN_ABS_RHO_MAX
                 and float(np.abs(rand_rhos).max()) < RANDOM_PER_DRAW_ABS_RHO_MAX)
    print(f"[gate] random anchor: mean|rho|={np.abs(rand_rhos).mean():.4f} "
          f"max|rho|={np.abs(rand_rhos).max():.4f} ok={random_ok}")
    if not identity_ok or not random_ok:
        sys.exit("GATE FAILED: the metric machinery cannot be trusted. Output not written.")

    arms = {
        "composite": {"what": "the shipped score_real_signals construction on pre-cutoff venues; "
                              "the arm under test", **arm_metrics(composite, y)},
        "raw_venue_count": {"what": "pre-cutoff venue count per bucket; THE NULL HYPOTHESIS",
                            **arm_metrics(raw_count, y)},
        "equal_weights": {"what": "the same four channel means at 0.25 each",
                          **arm_metrics(equal, y)},
        "tourist_zone_channel": {"what": "the pre-cutoff mean tourist-zone score alone",
                                 **arm_metrics(tz, y)},
        "random": rand_out,
    }
    for name in ("composite", "raw_venue_count", "equal_weights", "tourist_zone_channel"):
        a = arms[name]
        print(f"[arm] {name}: spearman={a['spearman_vs_outcome']} "
              f"p@50={a['precision_at_50']} lift_top={a['lift_top_decile']}")

    print(f"[boot] {N_BOOT} cell-clustered resamples")
    boot = cluster_bootstrap(buckets, composite, raw_count, y)
    lo, hi = boot["spearman_diff_composite_minus_density"]["interval_2p5_97p5"]
    if lo > 0:
        call = "composite_beats_density"
    elif hi < 0:
        call = "density_beats_composite"
    else:
        call = "not_resolved"
    print(f"[decision] diff={boot['spearman_diff_composite_minus_density']['point']} "
          f"interval=[{lo}, {hi}] call={call}")

    sentence = _required_sentence(call, {
        "n_buckets": n,
        "rho_composite": arms["composite"]["spearman_vs_outcome"],
        "rho_density": arms["raw_venue_count"]["spearman_vs_outcome"],
        "diff_point": boot["spearman_diff_composite_minus_density"]["point"],
        "diff_lo": lo, "diff_hi": hi,
    })

    import scipy  # noqa: F401 - version for the envelope
    versions = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "polars": pl.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    ws_path = repo / "results" / "whitespace.json"
    result = {
        "seed": SEED,
        "versions": versions,
        "generated_by": "scripts/whitespace_temporal.py --check-able",
        "data_sources": [
            we.DATA_SOURCE,
            {"name": "One Loop whitespace ranking under test (frozen, not modified)",
             "url": "results/whitespace.json produced by scripts/whitespace_exhibit.py",
             "sha256": we.sha256_file(ws_path)},
        ],
        "labels": ["real-public-data", "temporal holdout",
                   "outcome is venue formation, not merchant signing"],
        "what_this_is": ("the forward-looking check the whitespace head did not have: the "
                         "composite computed from venues existing before the cutoff, tested "
                         "against where card-accepting-universe venues actually appeared after "
                         "it, against a density null and pre-registered controls"),
        "pre_registration": PREREG,
        "design": {
            "cutoff_T": CUTOFF_T,
            "snapshot_S": SNAPSHOT_S,
            "outcome_window_months": 24,
            "unit": "the exhibit's own bucket: (cell_x, cell_y, category_group), grid 0.008 deg, "
                    "formed at >= 10 pre-cutoff venues",
            "construction_reuse": ("assign_group, compute_signals and make_buckets imported from "
                                   "scripts/whitespace_exhibit.py (not edited); frame construction "
                                   "refactored here to accept the date predicate; chain rule, "
                                   "KD-tree density and its p99 normalizer re-estimated within the "
                                   "pre-cutoff subset; tourist zones, MDR priors and the weights "
                                   "0.30/0.30/0.25/0.15 are constants and stay untouched"),
            "tie_break": "stable (cell_x, cell_y, category_group) order inside every top-k cut, "
                         "the shipped producer's own sort",
        },
        "n_pre_cutoff_universe_venues": None,   # filled below
        "n_buckets_pre_cutoff": n,
        "outcome_mean_per_bucket": round(float(y.mean()), 4),
        "outcome_nonzero_buckets": int((y > 0).sum()),
        "coverage": coverage,
        "gates": {
            "reproduction_of_shipped": repro,
            "identity": {"spearman": identity_rho, "tolerance": TOL, "ok": identity_ok},
            "random_anchor": {
                "mean_abs_spearman": round(float(np.abs(rand_rhos).mean()), 6),
                "max_abs_spearman": round(float(np.abs(rand_rhos).max()), 6),
                "mean_max": RANDOM_MEAN_ABS_RHO_MAX,
                "per_draw_max": RANDOM_PER_DRAW_ABS_RHO_MAX,
                "ok": random_ok,
            },
        },
        "arms": arms,
        "bootstrap": boot,
        "decision": {
            "rule_restated": PREREG["decision_rule"],
            "call": call,
        },
        "required_sentence": sentence,
        "caveats": CAVEATS,
        "check": {"command": "python3 scripts/whitespace_temporal.py --check",
                  "tolerance": TOL},
    }
    result["n_pre_cutoff_universe_venues"] = int(
        build_frame(data_path, CUTOFF_T).height)
    return result


def main():
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to fsq_sg.parquet")
    ap.add_argument("--out", default=str(repo / "results" / "whitespace_temporal.json"))
    ap.add_argument("--check", action="store_true",
                    help="recompute deterministically and compare to the committed JSON at 1e-6")
    args = ap.parse_args()

    data_path = we.find_data(args.data)
    got = we.sha256_file(data_path)
    if got != we.DATA_SOURCE["sha256"]:
        sys.exit(f"ERROR: sha256 mismatch for {data_path}: {got}")

    out_path = Path(args.out)
    saved = None
    if args.check:
        if not out_path.is_file():
            sys.exit(f"--check: {out_path} does not exist")
        saved = json.loads(out_path.read_text())

    result = build(data_path, repo)

    if args.check:
        diffs = we.compare_numeric(saved, result)
        if diffs:
            print("--check FAILED:", *diffs, sep="\n  ")
            sys.exit(1)
        print(f"--check OK: {out_path} reproduced numerically ({TOL})")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} B)")
    print("required_sentence:", result["required_sentence"])


if __name__ == "__main__":
    main()
