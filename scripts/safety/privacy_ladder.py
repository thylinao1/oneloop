#!/usr/bin/env python3
"""SAFE-B2, the disclosure-control ladder on the partner-facing whitespace ranking.

Same move as the leakage ladder (scripts/fm/ladder_eval.py): ONE frozen
partner-facing artifact, guards turned on one step at a time, and the price of
each guard printed beside it. Here the artifact is the whitespace ranked list
(results/whitespace.json) and the guards are disclosure controls rather than
evaluation protocols.

Rungs, along the declared spine:
  P0  raw output, no minimum cell frequency (n = 1), no contribution bound, no noise
  P1  small-cell suppression: minimum cell frequency n contributing POIs per cell
  P2  P1 plus per-entity contribution bounding on the density channel (degree bound d)
  P3  P2 plus calibrated Laplace noise on the published bucket score at a stated epsilon

Every utility metric is measured against P0 (the no-protection endpoint) AND against
the SHIPPED point (n = 10, no bound, no noise), because a partner receives the shipped
artifact today and the reader needs both references on the same axes.

WHAT THIS EXHIBIT PROTECTS, stated before any number: the privacy unit is one point of
interest, meaning one venue. It protects VENUES, not cardmembers. No cardmember data and
no American Express data enters this head at all, and the corpus is a public Foursquare
venue dataset. The sibling safety exhibits run on the public IBM TabFormer benchmark,
which is synthetic; this one runs on real public venue records. Either way the result
measures the mechanism, meaning what a disclosure guard of this shape costs an output of
this shape, and it is not a measurement of American Express's exposure.

Laptop, CPU, deterministic. --check recomputes every numeric leaf and compares at 1e-6.
"""
from __future__ import annotations

import os

# Cap threads BEFORE numeric imports (8GB shared machine; determinism).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "fm"))

import whitespace_exhibit as ws          # noqa: E402  the frozen producer we are pricing
from common import atomic_write_json, seed_everything, versions_dict  # noqa: E402

SEED = 42
N_SWEEP = [1, 3, 5, 10, 20, 50]          # minimum cell frequency (contributing POIs)
D_SWEEP = [5, 20, 100, None]             # contribution bound: max recipients per POI
EPS_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
NOISE_SEEDS = 20
CONTROL_SEEDS = 20
SHIPPED_N = ws.MIN_BUCKET_POIS           # 10
RELEASED_ROWS = ws.MAX_RANKING_ROWS      # 400
SPINE_N = SHIPPED_N                      # the spine holds n at the shipped value from P1 on
SPINE_D = 20                             # the spine's chosen contribution bound
SPINE_EPS = 10.0                         # the spine's named operating point for noise
NEIGH_CHUNK = 5_000


# --------------------------------------------------------------- signals ----

def geo_xy(df: pl.DataFrame) -> np.ndarray:
    """Equirectangular metres, identical to scripts/whitespace_exhibit.py:332-337."""
    lat = df["latitude"].to_numpy().astype(np.float64)
    lon = df["longitude"].to_numpy().astype(np.float64)
    lat0 = 1.3521
    x = (lon * math.cos(math.radians(lat0)) * ws.M_PER_DEG_LAT).astype(np.float64)
    y = (lat * ws.M_PER_DEG_LAT).astype(np.float64)
    return np.column_stack([x, y])


def neigh_unbounded(pts: np.ndarray) -> np.ndarray:
    """The shipped density signal: every POI within 250 m contributes, no bound.

    Byte-identical path to scripts/whitespace_exhibit.py:339-341."""
    tree = cKDTree(pts)
    n = tree.query_ball_point(pts, r=ws.DENSITY_RADIUS_M, workers=1, return_length=True)
    return np.asarray(n, dtype=np.int64) - 1  # exclude self


def neigh_degree_bounded(pts: np.ndarray, place_rank: np.ndarray, d: int) -> dict:
    """Per-entity contribution bounding on the density channel.

    Each POI contributes to at most d recipients: its d nearest neighbours inside
    250 m, ties broken by distance then by fsq_place_id (the pre-registered rule).
    The count for POI i is the number of POIs that selected i.

    The selection is resolved over the COMPLETE 250 m neighbourhood of every POI, not
    over a nearest-k candidate pool. A first pass of this code used a pool of d + 2 and
    the pool was full for 144,002 of the 146,048 POIs at d = 5, which left the choice
    among distance-tied candidates at the cut decided by the order the spatial index
    happened to return rather than by the stated rule. That is a reproducibility defect
    rather than a wrong answer, and it is fixed here rather than declared.

    Also returns the in-degree distribution, which is the measured size of the channel
    the adjacency invariant declares away.
    """
    n = len(pts)
    tree = cKDTree(pts)
    indeg = np.zeros(n, dtype=np.int64)
    outdeg = np.zeros(n, dtype=np.int64)
    ties_at_cut = 0
    for c0 in range(0, n, NEIGH_CHUNK):
        c1 = min(c0 + NEIGH_CHUNK, n)
        lists = tree.query_ball_point(pts[c0:c1], r=ws.DENSITY_RADIUS_M, workers=1)
        lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=c1 - c0)
        flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
        row = np.repeat(np.arange(c0, c1, dtype=np.int64), lens)
        keep = flat != row                       # drop self
        flat, row = flat[keep], row[keep]
        dist = np.hypot(pts[flat, 0] - pts[row, 0], pts[flat, 1] - pts[row, 1])
        order = np.lexsort((place_rank[flat], dist, row))
        flat, row, dist = flat[order], row[order], dist[order]
        cnt = np.bincount(row - c0, minlength=c1 - c0)
        starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        pos = np.arange(len(row), dtype=np.int64) - starts[row - c0]
        sel = pos < d
        np.add.at(indeg, flat[sel], 1)
        outdeg[c0:c1] = np.minimum(cnt, d)
        cut = np.flatnonzero(cnt > d)            # the cut falls inside this row
        if cut.size:
            last_in = dist[starts[cut] + d - 1]
            first_out = dist[starts[cut] + d]
            ties_at_cut += int(np.sum(last_in == first_out))
    return {"neigh": indeg, "outdeg": outdeg, "ties_at_cut": ties_at_cut}


def poi_scores(df: pl.DataFrame, pts: np.ndarray, neigh: np.ndarray,
               p99_override: float | None = None) -> dict:
    """Per-POI score, identical arithmetic to scripts/whitespace_exhibit.py:342-364,
    with the density neighbour array injected so the ladder can vary it."""
    lat0 = 1.3521
    x, y = pts[:, 0], pts[:, 1]
    p99 = float(np.percentile(neigh, 99)) if p99_override is None else float(p99_override)
    dens_norm = np.minimum(np.log1p(neigh) / math.log1p(p99), 1.0).astype(np.float32)

    tz = np.zeros(len(df), dtype=np.float32)
    for z in ws.TOURIST_ZONES:
        zx = z["lon"] * math.cos(math.radians(lat0)) * ws.M_PER_DEG_LAT
        zy = z["lat"] * ws.M_PER_DEG_LAT
        d_km = np.hypot(x - zx, y - zy) / 1000.0
        tz = np.maximum(tz, np.exp(-d_km / ws.TZ_DECAY_KM).astype(np.float32))

    priors = np.array([ws.MDR_PRIORS[g]["prior"] for g in df["category_group"].to_list()],
                      dtype=np.float32)
    indep = (~df["is_chain"].to_numpy()).astype(np.float32)
    score = (ws.WEIGHTS["mdr_sensitivity_prior"] * priors
             + ws.WEIGHTS["tourist_zone_proximity"] * tz
             + ws.WEIGHTS["local_density"] * dens_norm
             + ws.WEIGHTS["independent_share"] * indep).astype(np.float32)
    return {"score": score, "p99": p99}


# --------------------------------------------------------------- buckets ----

def make_buckets(df: pl.DataFrame, poi_score: np.ndarray, min_n: int) -> dict:
    """Grid cell x category group, mean of member POI scores, minimum cell frequency.

    Mirrors scripts/whitespace_exhibit.py:389-406 with the threshold parameterised.
    Bucket identity is (cell_x, cell_y, category_group), which is stable across
    rungs and independent of the label disambiguation the shipped script applies.
    """
    d2 = df.with_columns(
        pl.Series("poi_score", poi_score),
        (pl.col("latitude") // ws.GRID_DEG).cast(pl.Int32).alias("cell_y"),
        (pl.col("longitude") // ws.GRID_DEG).cast(pl.Int32).alias("cell_x"),
    )
    agg = (d2.group_by(["cell_x", "cell_y", "category_group"])
           .agg(pl.len().alias("n_pois"), pl.col("poi_score").mean().alias("score"))
           .filter(pl.col("n_pois") >= min_n)
           .sort(["cell_x", "cell_y", "category_group"]))
    keys = [f"{cx}|{cy}|{g}" for cx, cy, g in zip(
        agg["cell_x"].to_list(), agg["cell_y"].to_list(), agg["category_group"].to_list())]
    return {"keys": np.array(keys, dtype=object),
            "score": agg["score"].to_numpy().astype(np.float64),
            "n_pois": agg["n_pois"].to_numpy().astype(np.int64)}


def order_keys(keys: np.ndarray, score: np.ndarray) -> list[str]:
    """Released ordering: score descending, ties resolved by the stable
    (cell_x, cell_y, category_group) order, exactly as the shipped script does."""
    return [str(k) for k in keys[np.argsort(-score, kind="stable")]]


# --------------------------------------------------------------- utility ----

def utility(ref: list[str], rung: list[str]) -> dict:
    """Product metrics, not a generic information-loss score.

    spearman_full is over the buckets published at BOTH rungs, using each rung's own
    full ordering. The top-k metrics are on the released list (first 400 rows), which
    is what a partner actually acts on. top20_churn is the headline.
    """
    rr = {k: i for i, k in enumerate(rung)}
    common = [k for k in ref if k in rr]
    fr = {k: i for i, k in enumerate(ref)}
    rho_full = (float(spearmanr([fr[k] for k in common], [rr[k] for k in common]).statistic)
                if len(common) >= 3 else None)
    ref_rel, rung_rel = ref[:RELEASED_ROWS], rung[:RELEASED_ROWS]
    rr_rel = {k: i for i, k in enumerate(rung_rel)}
    c100 = [k for k in ref_rel[:100] if k in rr_rel]
    rho_100 = (float(spearmanr([fr[k] for k in c100], [rr_rel[k] for k in c100]).statistic)
               if len(c100) >= 3 else None)
    t20r, t20k = set(ref_rel[:20]), set(rung_rel[:20])
    return {
        "spearman_full": rho_full,
        "n_buckets_compared_full": len(common),
        "spearman_top100": rho_100,
        "n_top100_still_published": len(c100),
        "top100_overlap": len(set(ref_rel[:100]) & set(rung_rel[:100])),
        "top400_overlap": len(set(ref_rel) & set(rung_rel)),
        "top20_left": len(t20r - t20k),
        "top20_entered": len(t20k - t20r),
        "top20_churn": len(t20r - t20k) + len(t20k - t20r),
    }


UTIL_KEYS = ["spearman_full", "spearman_top100", "top100_overlap",
             "top400_overlap", "top20_churn"]


def spread(vals: list) -> dict:
    """Mean and percentile 95% spread across seeds. A degenerate spread is null and
    the renderer prints 'no interval'; no claim may be made from that row."""
    a = np.array([v for v in vals if v is not None], dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "spread": None, "n_seeds": 0}
    if a.size < 2:
        return {"mean": float(a[0]), "spread": None, "n_seeds": int(a.size)}
    return {"mean": float(a.mean()),
            "spread": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            "n_seeds": int(a.size)}


# ----------------------------------------------------------- sensitivity ----

def l1_sensitivity(min_n: int, d: int | None, p99: float) -> dict:
    """L1 sensitivity of the published bucket-score vector to adding or removing one POI.

    Derived from the bound, not assumed. Channels, each traced to a line of the
    shipped producer:

      own bucket   scripts/whitespace_exhibit.py:396, the bucket score is an unweighted
                   mean of per-POI scores that are bounded in [0, 1] by construction
                   (:361-364). Removing one member of a cell of n POIs moves the mean by
                   at most 1/(n-1), and the conditioning holds the published set fixed,
                   so the post-removal count is at least min_n. Bound 1/min_n.
      density      :339-343, the 250 m KD-tree. Under the contribution bound one POI is
                   a contributor to at most d recipients. One contributor moves a
                   recipient's log1p-normalised density by at most log(2)/log1p(p99),
                   which enters the score with weight 0.25 and is diluted by that
                   recipient's own cell size. Bound 0.25 * d * log(2)/log1p(p99) / min_n.
      chain rule   :327, a POI is a chain when its normalised name occurs at least 3
                   times. Removing one POI can flip at most 2 others from chain to
                   independent, each moving the score by the 0.15 independent weight.
                   Bound 2 * 0.15 / min_n.

    Returns run=False with the reason when d is unbounded, because then the a-priori
    bound is the dataset size rather than a constant and no epsilon may be quoted.
    """
    if d is None:
        return {"run": False,
                "reason": ("no contribution bound is declared, so one POI can contribute to "
                           "every POI within 250 m and the a-priori bound on the density "
                           "channel is the dataset size rather than a constant. This is why "
                           "contribution bounding comes before noise, and it is why no "
                           "epsilon is quoted at any unbounded rung."),
                "l1_sensitivity": None}
    dens_step = math.log(2.0) / math.log1p(p99)
    own = 1.0 / min_n
    dens = ws.WEIGHTS["local_density"] * d * dens_step / min_n
    chain = (ws.CHAIN_MIN_COUNT - 1) * ws.WEIGHTS["independent_share"] / min_n
    return {
        "run": True,
        "l1_sensitivity": own + dens + chain,
        "terms": {"own_bucket_mean": own, "density_channel": dens, "chain_rule_flip": chain},
        "inputs": {"min_cell_frequency": min_n, "contribution_bound_d": d,
                   "density_p99_normalizer": p99,
                   "max_density_step_per_contributor": dens_step},
    }


# ------------------------------------------------------------------- run ----

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(data_path: Path, shipped_path: Path) -> dict:
    t0 = time.time()
    seed_everything(SEED)
    df, n_universe = ws.build_poi_frame(data_path)
    pts = geo_xy(df)
    n_poi = df.height
    pid = df["fsq_place_id"].to_list()
    place_rank = np.argsort(np.argsort(np.array(pid, dtype=object).astype(str),
                                       kind="stable"), kind="stable").astype(np.int64)
    print(f"[ladder] {n_poi:,} card-accepting POIs of {n_universe:,} in the SG slice "
          f"({time.time() - t0:.1f}s)", flush=True)

    # --- the density signal at every contribution bound -----------------------
    dens = {}
    neigh0 = neigh_unbounded(pts)
    s0 = poi_scores(df, pts, neigh0)
    dens["unbounded"] = {"neigh": neigh0, "score": s0["score"], "p99": s0["p99"],
                         "audit": {"selection_is_exact": True, "ties_resolved_at_the_cut": 0,
                                   "max_recipients_per_poi": int(neigh0.max()),
                                   "mean_recipients_per_poi": float(neigh0.mean()),
                                   "note": "no bound: a POI contributes to every POI within 250 m"}}
    print(f"[ladder] density unbounded: p99={s0['p99']:.1f} max_degree={int(neigh0.max())} "
          f"({time.time() - t0:.1f}s)", flush=True)
    for d in [x for x in D_SWEEP if x is not None]:
        r = neigh_degree_bounded(pts, place_rank, d)
        s = poi_scores(df, pts, r["neigh"])
        dens[str(d)] = {"neigh": r["neigh"], "score": s["score"], "p99": s["p99"],
                        "audit": {"selection_is_exact": True,
                                  "ties_resolved_at_the_cut": r["ties_at_cut"],
                                  "max_out_degree": int(r["outdeg"].max()),
                                  "mean_out_degree": float(r["outdeg"].mean()),
                                  "max_in_degree": int(r["neigh"].max()),
                                  "p99_in_degree": float(np.percentile(r["neigh"], 99)),
                                  "note": ("out-degree is the bounded quantity and is what the "
                                           "sensitivity uses. in-degree is reported because the "
                                           "adjacency invariant declares the reselection channel "
                                           "away, and this is its measured size. "
                                           "ties_resolved_at_the_cut counts POIs whose d-th and "
                                           "(d+1)-th nearest neighbours sit at an identical "
                                           "distance, which the fsq_place_id tie-break resolves "
                                           "over the complete neighbourhood.")}}
        print(f"[ladder] density d={d}: p99={s['p99']:.1f} max_in_degree={int(r['neigh'].max())} "
              f"ties_at_cut={r['ties_at_cut']} ({time.time() - t0:.1f}s)", flush=True)
    # The bound and the p99 renormalizer move together at P2. This arm holds the
    # normalizer at the unbounded value so the reader can see both halves. It isolates
    # neither cleanly, and it is labelled a diagnostic rather than a rung for that reason.
    s_fixed = poi_scores(df, pts, dens[str(SPINE_D)]["neigh"], p99_override=s0["p99"])

    # --- P0, the no-protection endpoint, and the shipped point ----------------
    b_p0 = make_buckets(df, dens["unbounded"]["score"], 1)
    ord_p0 = order_keys(b_p0["keys"], b_p0["score"])
    b_ship = make_buckets(df, dens["unbounded"]["score"], SHIPPED_N)
    ord_ship = order_keys(b_ship["keys"], b_ship["score"])
    print(f"[ladder] P0 publishes {len(ord_p0):,} buckets; shipped point publishes "
          f"{len(ord_ship):,}", flush=True)

    def cell_stats(b: dict) -> dict:
        return {"n_buckets_published": int(len(b["keys"])),
                "n_rows_released": int(min(len(b["keys"]), RELEASED_ROWS)),
                "achieved_min_cell_frequency": int(b["n_pois"].min()),
                "median_cell_frequency": float(np.median(b["n_pois"])),
                "max_cell_frequency": int(b["n_pois"].max()),
                "n_singleton_cells": int((b["n_pois"] == 1).sum())}

    def rung_entry(rung_id, name, guard, n, d, eps, b, orderings, deterministic):
        """orderings: list of released orderings (one per seed for noisy rungs)."""
        u_p0 = [utility(ord_p0, o) for o in orderings]
        u_sh = [utility(ord_ship, o) for o in orderings]
        if deterministic:
            vs_p0, vs_ship = u_p0[0], u_sh[0]
        else:
            vs_p0 = {k: spread([u[k] for u in u_p0]) for k in UTIL_KEYS}
            vs_ship = {k: spread([u[k] for u in u_sh]) for k in UTIL_KEYS}
        return {
            "rung": rung_id, "name": name, "guard_added": guard,
            "config": {"min_cell_frequency": n, "contribution_bound_d": d, "epsilon": eps},
            "deterministic": deterministic,
            "cells": cell_stats(b),
            "utility_vs_p0": vs_p0,
            "utility_vs_shipped": vs_ship,
        }

    rungs = []

    # P0
    rungs.append(rung_entry("P0", "raw output, no protection",
                            "none", 1, None, None, b_p0, [ord_p0], True))
    # P1 sweep over the minimum cell frequency
    p1_by_n, control_by_n = {}, {}
    for n in N_SWEEP:
        b = make_buckets(df, dens["unbounded"]["score"], n)
        o = order_keys(b["keys"], b["score"])
        e = rung_entry("P1", "small-cell suppression", "minimum cell frequency",
                       n, None, None, b, [o], True)
        e["is_shipped_point"] = (n == SHIPPED_N)
        p1_by_n[str(n)] = e
        # no-model control: remove the same NUMBER of buckets at random from P0
        k_removed = len(ord_p0) - len(o)
        if k_removed > 0:
            u_p0, cells = [], []
            for s in range(CONTROL_SEEDS):
                rng = np.random.default_rng(SEED * 1000 + n * 10 + s)
                keep = np.ones(len(b_p0["keys"]), dtype=bool)
                keep[rng.choice(len(b_p0["keys"]), size=k_removed, replace=False)] = False
                oc = order_keys(b_p0["keys"][keep], b_p0["score"][keep])
                u_p0.append(utility(ord_p0, oc))
                cells.append(int(b_p0["n_pois"][keep].min()))
            control_by_n[str(n)] = {
                "arm": "random suppression of the same number of buckets",
                "n_buckets_removed": int(k_removed),
                "seeds": CONTROL_SEEDS,
                "achieved_min_cell_frequency_mean": float(np.mean(cells)),
                "utility_vs_p0": {k: spread([u[k] for u in u_p0]) for k in UTIL_KEYS},
            }
    # P2 sweep over the contribution bound, at the spine's n
    p2_by_d = {}
    for d in D_SWEEP:
        key = "unbounded" if d is None else str(d)
        b = make_buckets(df, dens[key]["score"], SPINE_N)
        o = order_keys(b["keys"], b["score"])
        e = rung_entry("P2", "contribution bounding on the density channel",
                       "max recipients per POI", SPINE_N, d, None, b, [o], True)
        e["density_audit"] = dens[key]["audit"]
        e["density_p99_normalizer"] = dens[key]["p99"]
        e["sensitivity"] = l1_sensitivity(SPINE_N, d, dens[key]["p99"])
        p2_by_d[key] = e
    # P3 sweep over epsilon, at the spine's n and d
    b_spine = make_buckets(df, dens[str(SPINE_D)]["score"], SPINE_N)
    sens = l1_sensitivity(SPINE_N, SPINE_D, dens[str(SPINE_D)]["p99"])
    p3_by_eps = {}
    for eps in EPS_SWEEP:
        scale = sens["l1_sensitivity"] / eps
        orderings = []
        for s in range(NOISE_SEEDS):
            rng = np.random.default_rng(SEED * 100_000 + int(eps * 100) * 100 + s)
            noisy = b_spine["score"] + rng.laplace(0.0, scale, size=len(b_spine["score"]))
            orderings.append(order_keys(b_spine["keys"], noisy))
        e = rung_entry("P3", "calibrated differential-privacy noise", "Laplace noise",
                       SPINE_N, SPINE_D, eps, b_spine, orderings, False)
        e["noise"] = {"mechanism": "Laplace", "scale": scale,
                      "l1_sensitivity": sens["l1_sensitivity"], "delta": 0.0,
                      "seeds": NOISE_SEEDS}
        p3_by_eps[f"{eps:g}"] = e
        print(f"[ladder] P3 eps={eps:g}: scale={scale:.4f} "
              f"top20_churn={e['utility_vs_shipped']['top20_churn']['mean']:.1f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    # --- why the suppression rung lands where it lands ------------------------
    p0_np = {k: int(v) for k, v in zip(b_p0["keys"], b_p0["n_pois"])}
    def head_cells(m):
        v = np.array([p0_np[k] for k in ord_p0[:m]], dtype=np.int64)
        return {"n": int(m), "min_contributing_pois": int(v.min()),
                "median_contributing_pois": float(np.median(v)),
                "share_below_shipped_threshold": float((v < SHIPPED_N).mean()),
                "share_below_50": float((v < 50).mean())}
    b_fx = make_buckets(df, s_fixed["score"], SPINE_N)
    o_fx = order_keys(b_fx["keys"], b_fx["score"])
    diagnostics = {
        "why_the_suppression_rung_costs_nothing_at_the_head": {
            "what_was_measured": (
                "the number of contributing venues behind each cell at the head of the raw P0 "
                "ranking. A minimum cell frequency can only remove a published row, so it can only "
                "cost the head if the head is made of small cells."),
            "p0_head": {str(m): head_cells(m) for m in (20, 100, 400)},
            "reading": (
                "the head of the raw ranking is made of large dense cells, so the frequency rule "
                "never reaches it. That is the mechanism behind the null, and it is a property of "
                "this scoring function rather than a general result: density is one of the four "
                "signals, so cells with many venues score high by construction."),
        },
        "spearman_is_degenerate_on_a_pure_suppression_rung": (
            "suppression removes rows and changes no surviving row's score, so the induced order "
            "over the buckets published at both rungs is identical and the rank correlation is 1 "
            "by construction, not by measurement. The random-suppression control returns the same "
            "1, which is the proof of the degeneracy rather than a second finding. Read the "
            "overlap and churn numbers on the suppression rungs and ignore spearman_full there."),
        "p2_with_the_p99_normalizer_held_fixed": {
            "what_this_is": (
                f"the contribution bound at d = {SPINE_D} with the density normalizer held at the "
                "unbounded p99 instead of recomputed. At P2 the bound and the renormalizer move "
                "together, and this arm shows the second half. It isolates neither cleanly and is "
                "a diagnostic, not a rung."),
            "p99_used": s0["p99"],
            "utility_vs_shipped": utility(ord_ship, o_fx),
        },
    }

    # --- guard prices along the spine ----------------------------------------
    spine = [
        ("P0", rungs[0]),
        ("P1", p1_by_n[str(SPINE_N)]),
        ("P2", p2_by_d[str(SPINE_D)]),
        ("P3", p3_by_eps[f"{SPINE_EPS:g}"]),
    ]
    prices, prices_by_key = [], {}
    for (fid, fr), (tid, tr) in zip(spine[:-1], spine[1:]):
        paired = (fr["cells"]["n_buckets_published"] == tr["cells"]["n_buckets_published"])
        for metric in UTIL_KEYS:
            a, b = fr["utility_vs_p0"][metric], tr["utility_vs_p0"][metric]
            a_v = a["mean"] if isinstance(a, dict) else a
            b_v = b["mean"] if isinstance(b, dict) else b
            if a_v is None or b_v is None:
                continue
            ent = {
                "guard": tr["guard_added"], "from": fid, "to": tid, "metric": metric,
                "value_before": a_v, "value_after": b_v, "price": b_v - a_v,
                "paired": bool(paired),
            }
            if not paired:
                ent["note"] = ("this step changes WHICH buckets are published, so the two rungs "
                               "are not scored on identical units. No interval exists for this "
                               "price and no renderer or copy slot may present it as if it had one.")
            elif tr["deterministic"]:
                ent["note"] = "deterministic, no interval needed"
            else:
                ent["note"] = ("the mechanism is stochastic. The spread below is across "
                               f"{NOISE_SEEDS} noise seeds on identical units, which is mechanism "
                               "randomness and not sampling uncertainty. The rung before this one "
                               "is deterministic, so the price spread is the value spread shifted "
                               "by that constant.")
                ent["value_after_spread_across_seeds"] = b["spread"]
                ent["price_spread_across_seeds"] = ([b["spread"][0] - a_v, b["spread"][1] - a_v]
                                                    if b["spread"] is not None else None)
            prices.append(ent)
            prices_by_key[f"{fid}_{tid}_{metric}"] = ent

    # --- the rule against its no-model control, stated whichever way it lands --
    comparisons, comparisons_by_key = [], {}
    for n, ctl in control_by_n.items():
        for metric in ("top20_churn", "top100_overlap", "top400_overlap"):
            rule_v = p1_by_n[n]["utility_vs_p0"][metric]
            c = ctl["utility_vs_p0"][metric]
            better_is_low = (metric == "top20_churn")
            sp = c["spread"]
            if sp is None:
                direction = "no_interval"
            elif (rule_v < sp[0] if better_is_low else rule_v > sp[1]):
                direction = "rule_wins"
            elif (rule_v > sp[1] if better_is_low else rule_v < sp[0]):
                direction = "control_wins"
            else:
                direction = "not_separated"
            ent = {
                "a": f"minimum cell frequency rule at n = {n}",
                "b": "random suppression of the same number of buckets",
                "metric": metric, "a_value": rule_v, "b_value": c["mean"],
                "b_spread_across_seeds": sp, "difference": rule_v - c["mean"],
                "direction": direction,
                "note": ("the control removes the same VOLUME by a rule that ignores cell size, so "
                         "the difference is what the targeting buys and nothing else. A direction "
                         "of control_wins or not_separated would mean the frequency rule's "
                         "targeting is buying nothing, and it would ship in those words."),
            }
            comparisons.append(ent)
            comparisons_by_key[f"n{n}_{metric}"] = ent

    # --- does P0 reproduce the shipped artifact ------------------------------
    shipped = json.loads(shipped_path.read_text())
    ship_rank = shipped["ranking"]
    idx_of = {str(k): i for i, k in enumerate(b_ship["keys"])}
    got = [round(float(b_ship["score"][idx_of[k]]), 6) for k in ord_ship[:len(ship_rank)]]
    want = [float(r["score_real_signals"]) for r in ship_rank]
    max_abs = max(abs(g - w) for g, w in zip(got, want))
    repro = {
        "compared": "score_real_signals of the shipped 400 released rows, in released order",
        "n_rows": len(want), "max_abs_diff": max_abs, "tolerance": 1e-6,
        "ok": bool(max_abs <= 1e-6),
        "n_buckets_shipped": int(shipped["n_buckets"]),
        "n_buckets_recomputed_at_shipped_threshold": int(len(b_ship["keys"])),
        "note": ("the ladder recomputes the shipped artifact from source before it prices "
                 "anything, so the rung it calls 'shipped' is the artifact a partner receives "
                 "and not a lookalike"),
    }
    if not repro["ok"] or repro["n_buckets_shipped"] != repro["n_buckets_recomputed_at_shipped_threshold"]:
        sys.exit(f"ERROR: the ladder does not reproduce results/whitespace.json: {repro}")

    # --- sanity, the calibration control on the measurement pipeline ----------
    ord_id = order_keys(b_p0["keys"], b_p0["score"])
    ident = utility(ord_p0, ord_id)
    big_eps = 1e6
    rng = np.random.default_rng(SEED)
    nz = b_spine["score"] + rng.laplace(0.0, sens["l1_sensitivity"] / big_eps,
                                        size=len(b_spine["score"]))
    ord_nz = order_keys(b_spine["keys"], nz)
    ord_spine = order_keys(b_spine["keys"], b_spine["score"])
    near = utility(ord_spine, ord_nz)
    sanity = {
        "what_this_is": ("the measurement pipeline has to be able to report both no change and "
                         "total change, or a small price is not evidence of anything"),
        "identity_rung": {"config": "P0 scored against itself", **ident},
        "vanishing_noise_rung": {"config": f"P3 at epsilon {big_eps:g}, one seed", **near},
        "destroyed_rung": {"config": f"P3 at epsilon {EPS_SWEEP[0]:g}",
                           "top20_churn_vs_shipped":
                               p3_by_eps[f"{EPS_SWEEP[0]:g}"]["utility_vs_shipped"]["top20_churn"],
                           "spearman_full_vs_shipped":
                               p3_by_eps[f"{EPS_SWEEP[0]:g}"]["utility_vs_shipped"]["spearman_full"]},
    }

    # ------------------------------------------------------------- envelope --
    import scipy  # noqa: F401
    versions = versions_dict()
    versions["scipy"] = scipy.__version__

    out = {
        "seed": SEED,
        "versions": versions,
        "generated_by": "scripts/safety/privacy_ladder.py --check-able",
        # ws.DATA_SOURCE's name carries an em dash, which the tone lock forbids anywhere the
        # page can render it. Same url and same sha256, so provenance is unchanged and the
        # entry still resolves to the identical file; only the punctuation of the label moves.
        # The upstream constant is left alone on purpose: it is also the label inside the
        # already shipped results/whitespace.json, and rewriting it there would mean
        # regenerating a frozen exhibit.
        "data_sources": [dict(ws.DATA_SOURCE, name=ws.DATA_SOURCE["name"].replace(" — ", ", ")),
                         {"name": "One Loop whitespace ranking (the rung-zero artifact under test)",
                          "url": "generated by scripts/whitespace_exhibit.py -> results/whitespace.json",
                          "sha256": sha256_file(shipped_path)}],
        "labels": ["real public POI data", "no cardmember data", "pseudonymized",
                   "mechanism-not-exposure", "disclosure-control ladder"],

        "what_this_is": (
            "The disclosure-control ladder on the partner-facing whitespace ranking. One frozen "
            "artifact, disclosure guards turned on one step at a time, and the price of each guard "
            "measured on the product metric rather than on a proxy. It is the same move as the "
            "leakage ladder, pointed at what leaves the perimeter instead of at how we score."),
        "mechanism_not_exposure": (
            "The privacy unit here is one point of interest, meaning one venue. This exhibit "
            "protects VENUES, not cardmembers: no cardmember data and no American Express data "
            "enters this head at all, and the corpus is the public Foursquare OS Places Singapore "
            "slice. The sibling safety exhibits run on the public IBM TabFormer benchmark, which "
            "is synthetic. Either way the result measures the mechanism, meaning what a guard of "
            "this shape costs an output of this shape. It is not a measurement of American "
            "Express's exposure and must never be presented as one."),
        "scope_limits": [
            "The guarantee covers the published bucket SCORES. The published contributing-POI "
            "count per cell is released exactly and is not protected by any rung here.",
            "The minimum-frequency filter is a data-dependent test on the released set. A "
            "threshold test of that kind is not itself differentially private, so the epsilon at "
            "P3 is conditional on the published set and is not a guarantee over set membership. "
            "A production release would need a private threshold test and noisy counts.",
            "The embedding column produced by the whitespace stage-two wire is out of scope here "
            "and carries no rung.",
            "The corridor head cannot carry this ladder at all: its smallest unit is a "
            "country-month count parsed from an already-published national statistic, so there is "
            "no contributing-entity count to expose, nothing to suppress, and no per-entity "
            "contribution to bound. That is a stated null, not an omission.",
        ],
        "privacy_unit": {
            "unit": "one point of interest (one venue)",
            "neighbouring_dataset": ("add or remove one POI from the Foursquare Singapore slice; "
                                     "all other POIs unchanged"),
            "who_this_protects": "venues, not cardmembers",
            "contributions_per_unit": ("one POI belongs to exactly one grid-cell-by-category "
                                       "bucket, and reaches other buckets only through the 250 m "
                                       "density channel, which is what the P2 bound limits"),
        },
        "declared_invariants": [
            {"invariant": "the 250 m contribution adjacency (which POI contributes to which)",
             "why": ("removing a POI makes every contributor that had selected it reselect a "
                     "replacement recipient, and that cascade is not bounded by the out-degree "
                     "bound. Holding the adjacency fixed is what makes the closed-form bound "
                     "hold, and it is declared here in the same way the 2020 Census published "
                     "its invariants rather than folding them into the epsilon."),
             "measured_size": "see the per-rung density_audit in_degree figures"},
            {"invariant": "the p99 density normalizer, published as computed",
             "why": ("it is a global order statistic over the whole slice, so it is not protected "
                     "by per-POI noise. Its value is recorded at every contribution bound.")},
            {"invariant": "the published bucket set, held fixed for the noise accounting",
             "why": "see scope_limits; the threshold test is not made private here"},
        ],
        "epsilon_reporting_rules_followed": {
            "unit": "epsilon per POI, per single release of the ranking",
            "delta": 0.0,
            "mechanism": "Laplace on each published bucket score",
            "composition": (
                "The quoted epsilon is for ONE release. Re-reading the published artifact costs "
                "nothing, because the same noisy numbers are served every time. Re-noising costs: "
                "under basic sequential composition k independent releases of the same ranking "
                "cost k times epsilon, so a partner-facing surface that regenerates the ranking on "
                "every query has no fixed budget. The deployment answer is to fix the release, not "
                "to re-noise per query, and that is a design constraint this ladder makes visible."),
            "not_comparable": (
                "This epsilon is not comparable to any other deployment's. A 2020 Census budget of "
                "19.61 per person per decade and an Exposure Notification local epsilon of 8 per "
                "client are different privacy units under different neighbouring-dataset "
                "definitions with different composition accounting. Quoting them side by side as "
                "if they were one currency is the error, not the comparison."),
            "not_a_claim_about_amex": (
                "This is a guarantee over a public venue dataset. It is not a claim about anything "
                "American Express holds, and no rung here is a claim of compliance with anything."),
        },
        "reference_lines_position_not_compliance": {
            "how_to_read": ("These are positions against published references. None of them is a "
                            "compliance claim, and none would be ours to make on a public venue "
                            "corpus."),
            "pdpc_selected_topics_3_9": (
                "minimum k-anonymity value of 5 with relevant safeguards for sharing with external "
                "parties, k of 3 internally with relevant internal controls (advisory guidelines, "
                "not legally binding, though PDPC is likely to take positions consistent with them)"),
            "pdpc_guide_to_basic_anonymisation_caveat": (
                "the same body's guide records that k-anonymity 'may not be suitable for all types "
                "of datasets or other complex use cases (e.g. longitudinal or transactional data "
                "where the same indirect identifiers may appear in multiple records)', and notes "
                "the limitation on attribute disclosure by homogeneity attack. Transaction data is "
                "the warned case. Quoting the threshold without this caveat would be selective "
                "quotation from one document."),
            "these_are_different_quantities": (
                "a contributing-POI count per cell is a minimum frequency rule from the official "
                "statistics tradition. It is not a k-anonymity equivalence class over "
                "quasi-identifiers. The two are related and are not the same quantity, so our "
                "position is reported against both traditions rather than against the more "
                "flattering one."),
            "official_statistics_minimum_frequency": (
                "normally 3; the UK ONS Secure Research Service uses 10 for individuals and 3 for "
                "properties or companies, and counts zeros"),
            "our_position": (f"the shipped artifact publishes cells of at least {SHIPPED_N} "
                             "contributing venues, and the sweep prices every other setting"),
            "secondary_suppression_not_done": (
                "official statistical disclosure control also requires secondary suppression so a "
                "suppressed cell cannot be recovered by differencing against published totals. "
                "This ladder publishes no totals to difference against, so the question does not "
                "arise for the shipped artifact, and no secondary suppression was implemented or "
                "tested here."),
            "publishing_our_own_parameters": (
                "statistical agencies keep concentration-rule parameters confidential because "
                "publishing them weakens the protection. We publish ours, because the corpus is a "
                "public venue dataset with no respondent to protect and because publishing the "
                "parameter is what makes the price checkable. A deployed control plane would not "
                "make the same choice, and that is a difference of setting rather than a "
                "contradiction."),
        },
        "method": {
            "artifact_under_test": "results/whitespace.json, the ranked list a partner receives",
            "bucket_identity": "(cell_x, cell_y, category_group), stable across rungs",
            "released_rows": RELEASED_ROWS,
            "utility_axis": (
                "the product metric, not a generic information-loss score. top20_churn is the "
                "headline because the top of the list is the decision a partner acts on. "
                "spearman_full is over the buckets published at both rungs; the top-k metrics are "
                "on the released list."),
            "top20_churn_definition": (
                "the count of buckets ENTERING plus the count LEAVING the top 20, so it runs from "
                "0 to 40 and it is twice the number of rows replaced. Both halves are always equal "
                "and both are reported separately as top20_entered and top20_left, so no reader "
                "has to infer the convention."),
            "references": {
                "p0": "raw output, minimum cell frequency 1, no contribution bound, no noise",
                "shipped": f"minimum cell frequency {SHIPPED_N}, no contribution bound, no noise",
            },
            "spine": {"n": SPINE_N, "d": SPINE_D, "epsilon": SPINE_EPS,
                      "why_this_operating_point": (
                          f"n stays at the shipped {SHIPPED_N} so the spine prices the two guards "
                          f"the product does not have yet; d of {SPINE_D} is the mid point of the "
                          f"sweep and the smallest bound whose utility price is still small; "
                          f"epsilon of {SPINE_EPS:g} is named as the operating point and the whole "
                          "sweep is published beside it so the reader sees the curve and not a point")},
            "sweeps": {"min_cell_frequency": N_SWEEP,
                       "contribution_bound_d": [str(d) if d is not None else "unbounded"
                                                for d in D_SWEEP],
                       "epsilon": EPS_SWEEP},
            "contribution_bound_rule": (
                "each POI contributes to at most d recipients: its d nearest neighbours inside "
                "250 m, ties broken by distance then by fsq_place_id. The count for a POI is the "
                "number of POIs that selected it."),
            "control": (
                "random suppression: remove the same NUMBER of buckets at random from P0 rather "
                "than by the frequency rule, 20 seeds, same utility metrics. If the rule costs the "
                "same as random removal of the same volume, the rule's targeting is buying nothing "
                "and that ships."),
        },
        "universe": {"n_pois_in_slice": int(n_universe),
                     "n_pois_card_accepting_universe": int(n_poi)},
        "rungs_by_id": {rid: e for rid, e in spine},
        "p0": rungs[0],
        "p1_by_n": p1_by_n,
        "control_random_suppression_by_n": control_by_n,
        "p2_by_d": p2_by_d,
        "p3_by_epsilon": p3_by_eps,
        "guard_prices": prices,
        "guard_prices_by_key": prices_by_key,
        "comparisons": comparisons,
        "comparisons_by_key": comparisons_by_key,
        "diagnostics": diagnostics,
        "findings_as_obtained": {
            "1_suppression_is_free_at_the_head_and_not_free_deeper": (
                "the minimum cell frequency guard costs nothing at the top of the list and "
                "something real further down. At the shipped threshold of 10 contributing venues "
                "it removes 1,783 of the 3,319 raw cells, the top 20 does not move at all, and 62 "
                "of the 400 released rows are lost. At 50 venues it removes 2,736 cells, the top 20 "
                "still does not move, and 175 of the 400 released rows are lost."),
            "2_the_control_says_the_targeting_is_doing_the_work": (
                "removing the same number of cells at random instead moves a mean top-20 churn of "
                "22.2, which is about 11 of the 20 rows replaced, and loses about 211 of the 400 "
                "released rows. The rule's figures are 0 and 62. The control is what makes the "
                "null in finding 1 informative: the measurement could have reported a large cost "
                "and did not."),
            "3_the_null_has_a_mechanism_and_it_is_this_scoring_function": (
                "the head of the raw ranking is made of large cells, minimum 113 contributing "
                "venues in the top 20, so a minimum frequency rule never reaches it. Density is "
                "one of the four scoring signals, so dense cells score high by construction. This "
                "is a property of this scorer, not a general result about suppression, and a "
                "scorer that did not reward density would not inherit the null."),
            "4_the_contribution_bound_is_where_the_price_shows_up": (
                "bounding one venue to at most 20 recipients gives a top-20 churn of 10, which is "
                "5 of the 20 rows replaced, and drops the full rank correlation against the "
                "shipped ranking to 0.9252. That is the guard the product does not have today, and "
                "it is the one that buys a finite sensitivity: without it the density channel has "
                "no a-priori bound and no epsilon can be quoted at all."),
            # Do NOT call any epsilon in this sweep meaningful. Epsilon is an exponent, the
            # cost is reported at every step, and the reader places the number.
            "5_the_noise_rung_is_expensive_at_every_epsilon_worth_quoting": (
                "the sweep runs from epsilon 0.5 to epsilon 100 per venue at delta 0, on top of "
                "the contribution bound, and the price is reported at every step rather than at "
                "one chosen step. At epsilon 10 the noise adds 7.7 to the top-20 churn, which is "
                "about 4 more of the 20 rows replaced, taking the total to 17.7. At epsilon 1 the "
                "top 20 is essentially replaced, at 38.4 of a possible 40. The curve only returns "
                "to the cost of the contribution bound on its own at epsilon 50 and above, where "
                "the churn is 10.7 and then 9.9 against the noiseless 10. We call none of these a "
                "guarantee, and we quote none of them as one. Epsilon is an exponent: a per-venue "
                "epsilon of 10 bounds the change in output probability from adding or removing "
                "one venue by e to the tenth, a factor of about 22,000, and the cheap end of this "
                "sweep is further out still. The numbers are here at every step so a reader can "
                "place them against whatever epsilon they would accept. The reading we defend is "
                "the narrow one: a ranking this short and this tightly packed at the head does "
                "not absorb calibrated noise well at any epsilon a privacy researcher would "
                "accept."),
            "what_this_does_not_say": (
                "no rung here is a compliance claim, and none of it transfers to American Express "
                "data. It measures what disclosure guards of this shape cost an output of this "
                "shape, on a public venue corpus, for a privacy unit that is a venue and not a "
                "cardmember."),
        },
        "deviations_from_preregistration": [
            {"deviation": "output file is results/safety_privacy_ladder.json",
             "prereg": "SAFETY-DESIGN.md SAFE-B2 named results/safety_ladder.json",
             "reason": "the implementation brief named the longer filename explicitly, and it is "
                       "the less collidable name beside the existing results/ladder.json"},
            {"deviation": "P0 is the raw output with no minimum cell frequency (n = 1), not the "
                          "shipped artifact",
             "prereg": "SAFETY-DESIGN.md SAFE-B2 put R0 at the shipped threshold of 10",
             "reason": "the brief requires P0 to be the raw unprotected output, and the reporting "
                       "rules require the no-protection endpoint on the same axes. Every metric is "
                       "therefore reported against BOTH P0 and the shipped point, so nothing the "
                       "pre-registration asked for is lost and the shipped point stays visible"},
            {"deviation": "n = 1 was added to the minimum cell frequency sweep",
             "prereg": "n in {3, 5, 10, 20, 50}",
             "reason": "n = 1 is the P0 endpoint; the five pre-registered values all ran unchanged"},
            {"deviation": "the contribution bound is resolved over the complete 250 m neighbourhood "
                          "rather than a nearest-k candidate pool",
             "prereg": "the pre-registered fallback was to declare the density channel open if the "
                       "degree bound could not be made deterministic",
             "reason": "a first pass used a pool of d + 2 and the pool was full for 144,002 of "
                       "146,048 POIs at d = 5, so the choice among distance-tied candidates at the "
                       "cut was decided by the spatial index rather than by the stated rule. "
                       "Resolving over the complete neighbourhood makes the rule exact, so the "
                       "fallback was not needed. It changed the d = 20 top-20 churn from 8 to 10, "
                       "which is why it was worth fixing rather than declaring"},
            {"deviation": "epsilon is reported under three declared invariants rather than "
                          "unconditionally",
             "prereg": "R3 runs only if a closed-form sensitivity can be derived, else run: false",
             "reason": "the closed form holds once the 250 m contribution adjacency, the p99 "
                       "density normalizer and the published bucket set are declared invariant. "
                       "All three are named in declared_invariants and in scope_limits with their "
                       "measured sizes, in the way the 2020 Census published its invariants rather "
                       "than folding them into the budget. Quoting the epsilon without them would "
                       "be the overclaim"},
        ],
        "r0_reproduces_shipped": repro,
        "sanity": sanity,
        "check": {"command": "python scripts/safety/privacy_ladder.py --check",
                  "tolerance": 1e-6,
                  "note": ("recomputes every rung from data/fsq_sg.parquet and compares every "
                           "numeric leaf against the committed results/safety_privacy_ladder.json. "
                           "CPU only and deterministic, so it needs no node pinning.")},
    }
    return out


# ----------------------------------------------------------------- check ----

def numeric_leaves(obj, prefix=""):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield prefix, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from numeric_leaves(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from numeric_leaves(v, f"{prefix}/{i}")


SKIP_CHECK_PREFIXES = ("/versions",)


def compare(fresh: dict, stored: dict, tol: float) -> int:
    a = {k: v for k, v in numeric_leaves(fresh) if not k.startswith(SKIP_CHECK_PREFIXES)}
    b = {k: v for k, v in numeric_leaves(stored) if not k.startswith(SKIP_CHECK_PREFIXES)}
    bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append((k, a.get(k), b.get(k)))
        elif not math.isclose(a[k], b[k], rel_tol=0.0, abs_tol=tol):
            bad.append((k, a[k], b[k]))
    if bad:
        for k, x, y in bad[:25]:
            print(f"CHECK MISMATCH {k}: recomputed {x} vs stored {y}")
        print(f"CHECK FAILED: {len(bad)} numeric leaves differ beyond {tol:g}")
        return 5
    print(f"CHECK OK: {len(a)} numeric leaves reproduce within {tol:g}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data" / "fsq_sg.parquet"))
    ap.add_argument("--shipped", default=str(REPO / "results" / "whitespace.json"))
    ap.add_argument("--out", default=str(REPO / "results" / "safety_privacy_ladder.json"))
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare every numeric leaf at 1e-6; exit 0/5")
    ap.add_argument("--check-tol", type=float, default=1e-6)
    args = ap.parse_args()

    data_path = Path(args.data)
    got = sha256_file(data_path)
    if got != ws.DATA_SOURCE["sha256"]:
        sys.exit(f"ERROR: sha256 mismatch for {data_path}: {got}")

    out = build(data_path, Path(args.shipped))

    if args.check:
        stored = json.loads(Path(args.out).read_text())
        return compare(out, stored, args.check_tol)

    atomic_write_json(args.out, out)
    print(f"\nwrote {args.out}")
    print(f"  P0 publishes {out['p0']['cells']['n_buckets_published']:,} buckets, "
          f"{out['p0']['cells']['n_singleton_cells']:,} of them single venues")
    for n in N_SWEEP:
        e = out["p1_by_n"][str(n)]
        u = e["utility_vs_p0"]
        c = out["control_random_suppression_by_n"].get(str(n))
        ctl = (f"  control churn {c['utility_vs_p0']['top20_churn']['mean']:.1f}"
               if c else "  control n/a")
        print(f"  P1 n={n:>2}: {e['cells']['n_buckets_published']:>6,} buckets, "
              f"rho_full {u['spearman_full']:.4f}, top20 churn {u['top20_churn']:>2}{ctl}")
    for k, e in out["p2_by_d"].items():
        u = e["utility_vs_shipped"]
        s = e["sensitivity"]
        print(f"  P2 d={k:>9}: rho_full vs shipped {u['spearman_full']:.4f}, "
              f"top20 churn {u['top20_churn']:>2}, "
              f"L1 sensitivity {s['l1_sensitivity'] if s['run'] else 'none (unbounded)'}")
    for k, e in out["p3_by_epsilon"].items():
        u = e["utility_vs_shipped"]
        print(f"  P3 eps={k:>5}: rho_full vs shipped {u['spearman_full']['mean']:.4f}, "
              f"top20 churn {u['top20_churn']['mean']:.1f} "
              f"{u['top20_churn']['spread']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
