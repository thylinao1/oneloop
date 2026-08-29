#!/usr/bin/env python3
"""One Loop: WS-B1 uplift exhibit (hero).

Produces results/uplift.json with the committed result envelope.

Pre-registered design (committed BEFORE the full-data run; seed=42):
  * Data: Criteo Uplift v2.1 (randomized 85/15; 13,979,592 rows; f0-f11,
    treatment, conversion, visit). Treatment propensity is a KNOWN randomized
    constant ~0.85 by design (Criteo v2.1 docs); we also record the measured
    share and use the measured training-split share in the X-learner blend.
  * Split: stratified by treatment arm, 70% train / 30% holdout, seed 42.
  * Rankings compared on the SAME holdout:
      (a) response-propensity targeting = LightGBM P(conversion|X) trained on
          the TREATED arm of the training split; rank holdout by predicted
          response;
      (b) X-learner uplift targeting (manual, LightGBM base learners):
          per-arm outcome models mu1/mu0 -> imputed effects
          D1 = Y - mu0(X) on treated, D0 = mu1(X) - Y on control ->
          effect regressions tau1/tau0 -> CATE = e*tau0 + (1-e)*tau1 with
          e = treatment propensity; rank holdout by predicted CATE.
  * Metrics: Qini curve + Qini coefficient for both rankings; uplift@{10,20,30}%.
    CIs: stratified bootstrap (resample rows WITHIN each treatment arm, same
    resample evaluates both rankings -> paired delta CI), B=200, percentile.
  * Primary outcome: conversion (~0.29% base rate). Robustness: visit.
  * Hillstrom (randomized 3-arm): Womens E-Mail vs No E-Mail; per-segment
    (recency buckets x history_segment) model rankings vs DIRECTLY measured
    randomized uplift (visit + spend); verdicts flag segments where the
    response model wastes budget.

Honesty rider: results reported as obtained, whichever direction.
Determinism: all randomness seeded; --check re-evaluates the holdout from the
saved prediction cache (and retrains the tiny single-thread Hillstrom models)
and asserts numerical identity at 1e-6.

Targeting block (added 2026-08-23, CONTRACT §2 key `criteo.targeting_at_k`):
  for the visit outcome and the conversion outcome separately, at k = 10/20/30%,
  the incremental outcome rate inside the targeted top-k under CATE ranking,
  the same under response ranking, the paired difference between the two, and
  the whole-holdout ATE as a reference line, every one with the stratified
  bootstrap CI already used above. It reuses the SAME bootstrap reps, so no new
  random draws are taken and no previously reported number moves.

Usage:
  python scripts/uplift_exhibit.py             # full run -> results/uplift.json
  python scripts/uplift_exhibit.py --dev       # 1M-row dev loop -> results/dev/
  python scripts/uplift_exhibit.py --check     # verify from cache, exit 0/1
  python scripts/uplift_exhibit.py --targeting # add criteo.targeting_at_k from cache
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
RESULTS = PROJ / "results"
CACHE_DIR = RESULTS / "cache"

CRITEO_GZ = DATA / "criteo-uplift-v2.1.csv.gz"
CRITEO_PARQUET = DATA / "criteo_f32.parquet"
HILLSTROM_CSV = DATA / "hillstrom.csv"
CHECKSUMS = DATA / "CHECKSUMS.txt"

SEED = 42
HOLDOUT_FRAC = 0.30
BOOTSTRAP_REPS = 200
UPLIFT_AT_KS = (0.10, 0.20, 0.30)
N_CURVE_POINTS = 200
TOL = 1e-6
TRAIN_THREADS = 4  # M2/8GB guard; --check and Hillstrom always use 1
FCOLS = [f"f{i}" for i in range(12)]

CRITEO_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
HILLSTROM_URL = (
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_"
    "E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

RECENCY_BUCKETS = [(1, 3), (4, 6), (7, 9), (10, 12)]


# ---------------------------------------------------------------- data layer
def ensure_parquet() -> None:
    """One-time stream conversion csv.gz -> float32 parquet (low peak RAM)."""
    import polars as pl

    if CRITEO_PARQUET.exists():
        return
    tmp = DATA / "_criteo_tmp.csv"
    print(f"[data] decompressing {CRITEO_GZ.name} -> {tmp.name} (3.2GB, once)")
    with gzip.open(CRITEO_GZ, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 24)
    print("[data] streaming cast -> parquet (float32 features, int8 labels)")
    (
        pl.scan_csv(tmp)
        .select(
            [pl.col(c).cast(pl.Float32) for c in FCOLS]
            + [
                pl.col("treatment").cast(pl.Int8),
                pl.col("conversion").cast(pl.Int8),
                pl.col("visit").cast(pl.Int8),
            ]
        )
        .sink_parquet(CRITEO_PARQUET, compression="zstd")
    )
    tmp.unlink()
    print(f"[data] wrote {CRITEO_PARQUET} ({CRITEO_PARQUET.stat().st_size/1e6:.0f} MB)")


def load_criteo(dev: bool):
    import polars as pl

    df = pl.scan_parquet(CRITEO_PARQUET).collect()
    if dev:
        # the raw file is ordered by treatment arm -> head() would be one-armed;
        # use a seeded uniform 1M-row sample instead
        rng = np.random.default_rng(SEED + 2)
        idx = np.sort(rng.choice(len(df), size=1_000_000, replace=False))
        df = df[idx]
    X = df.select(FCOLS).to_numpy()  # float32 (n, 12)
    t = df["treatment"].to_numpy()
    yc = df["conversion"].to_numpy()
    yv = df["visit"].to_numpy()
    n = len(df)
    del df
    return X, t.astype(np.int8), yc.astype(np.int8), yv.astype(np.int8), n


def read_checksums() -> dict[str, str]:
    out = {}
    for line in CHECKSUMS.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[Path(parts[1]).name] = parts[0]
    return out


# ------------------------------------------------------------------- models
def lgbm_params(objective: str, threads: int) -> dict:
    return dict(
        objective=objective,
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=63,
        min_child_samples=200,
        max_bin=255,
        subsample=1.0,
        colsample_bytree=1.0,
        deterministic=True,
        force_row_wise=True,
        seed=SEED,
        num_threads=threads,
        verbosity=-1,
    )


def fit_binary(X, y, threads):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**lgbm_params("binary", threads)).fit(X, y)


def fit_reg(X, y, threads):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**lgbm_params("regression", threads)).fit(X, y)


def x_learner_scores(X_tt, y_tt, X_ct, y_ct, X_ho, e, threads):
    """Manual X-learner on LightGBM. Returns (response_score, cate_score) on
    the holdout. response_score is mu1 (treated-arm response model); the
    response-propensity targeting baseline shares mu1 by construction.
    e = treatment propensity used in the blend tau = e*tau0 + (1-e)*tau1."""
    print(f"    [xl] mu1 (treated, n={len(X_tt):,})")
    mu1 = fit_binary(X_tt, y_tt, threads)
    print(f"    [xl] mu0 (control, n={len(X_ct):,})")
    mu0 = fit_binary(X_ct, y_ct, threads)
    d1 = y_tt.astype(np.float64) - mu0.predict_proba(X_tt)[:, 1]
    d0 = mu1.predict_proba(X_ct)[:, 1] - y_ct.astype(np.float64)
    print("    [xl] tau1 (imputed effects, treated)")
    tau1 = fit_reg(X_tt, d1, threads)
    print("    [xl] tau0 (imputed effects, control)")
    tau0 = fit_reg(X_ct, d0, threads)
    resp = mu1.predict_proba(X_ho)[:, 1]
    cate = e * tau0.predict(X_ho) + (1.0 - e) * tau1.predict(X_ho)
    return resp.astype(np.float32), cate.astype(np.float32)


# ------------------------------------------------------------------ metrics
def qini_arrays(order, t, y, w):
    """Cumulative weighted Qini quantities along a fixed descending-score
    order. Returns (phi, q_norm): population fraction targeted and incremental
    conversions per customer q(phi)/W. All accumulation in float64."""
    t_o = t[order].astype(np.float32)
    y_o = y[order].astype(np.float32)
    w_o = w[order].astype(np.float32)
    wt = w_o * t_o
    wc = w_o * (1.0 - t_o)
    cum_nt = np.cumsum(wt, dtype=np.float64)
    cum_nc = np.cumsum(wc, dtype=np.float64)
    cum_yt = np.cumsum(wt * y_o, dtype=np.float64)
    cum_yc = np.cumsum(wc * y_o, dtype=np.float64)
    W = cum_nt[-1] + cum_nc[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        q = cum_yt - cum_yc * np.where(cum_nc > 0, cum_nt / np.maximum(cum_nc, 1e-12), 0.0)
    phi = (cum_nt + cum_nc) / W
    return phi, q / W, cum_nt, cum_nc, cum_yt, cum_yc


def qini_coefficient(phi, q_norm):
    """Area between the Qini curve and the random-targeting diagonal.
    Units: incremental conversions per customer (not normalized by the
    perfect-model area)."""
    auc = float(np.trapezoid(q_norm, phi))
    random_area = 0.5 * float(q_norm[-1]) * float(phi[-1])
    return auc - random_area


def uplift_at_fraction(phi, cum_nt, cum_nc, cum_yt, cum_yc, k):
    """Treated-vs-control outcome-rate difference within the top-k fraction."""
    i = int(np.searchsorted(phi, k, side="left"))
    i = min(i, len(phi) - 1)
    rt = cum_yt[i] / max(cum_nt[i], 1e-12)
    rc = cum_yc[i] / max(cum_nc[i], 1e-12)
    return float(rt - rc)


def evaluate_ranking(order, t, y, w):
    phi, qn, cnt, cnc, cyt, cyc = qini_arrays(order, t, y, w)
    out = {"qini": qini_coefficient(phi, qn)}
    for k in UPLIFT_AT_KS:
        out[f"u@{int(k*100)}"] = uplift_at_fraction(phi, cnt, cnc, cyt, cyc, k)
    # Whole-holdout treated-minus-control difference (the overall ATE). Taken at
    # phi=1 so it is ranking-independent by construction, and it rides the same
    # bootstrap weights as the top-k numbers, which makes the ATE reference line
    # and the top-k estimates comparable rep for rep.
    out["ate"] = uplift_at_fraction(phi, cnt, cnc, cyt, cyc, 1.0)
    return out, (phi, qn)


def curve_points(phi, qn):
    idx = np.unique(np.linspace(0, len(phi) - 1, N_CURVE_POINTS).astype(np.int64))
    return (
        [round(float(v), 6) for v in phi[idx]],
        [round(float(v), 10) for v in qn[idx]],
    )


def holdout_evaluation(t_ho, outcomes: dict, scores: dict, boot_reps: int):
    """Full deterministic evaluation: point estimates + stratified paired
    bootstrap. scores = {ranking_name: score_vector}; outcomes = {name: y}.
    Returns (points, cis, delta_ci, curves, delta_k_ci). Single bootstrap
    resample per rep is shared across rankings AND outcomes (paired within-arm
    resampling), so every delta CI is paired."""
    n = len(t_ho)
    orders = {
        r: np.argsort(-s, kind="stable").astype(np.int64) for r, s in scores.items()
    }
    ones = np.ones(n, dtype=np.float32)
    points, curves = {}, {}
    for oc, y in outcomes.items():
        for r in orders:
            res, (phi, qn) = evaluate_ranking(orders[r], t_ho, y, ones)
            points[(oc, r)] = res
            curves[(oc, r)] = curve_points(phi, qn)

    idx_t = np.flatnonzero(t_ho == 1)
    idx_c = np.flatnonzero(t_ho == 0)
    rng = np.random.default_rng(SEED + 1)
    stats: dict = {k: [] for k in points}
    delta_qini: dict = {oc: [] for oc in outcomes}
    t0 = time.time()
    for b in range(boot_reps):
        w = np.zeros(n, dtype=np.float32)
        draw_t = rng.integers(0, len(idx_t), len(idx_t))
        draw_c = rng.integers(0, len(idx_c), len(idx_c))
        w[idx_t] = np.bincount(draw_t, minlength=len(idx_t)).astype(np.float32)
        w[idx_c] = np.bincount(draw_c, minlength=len(idx_c)).astype(np.float32)
        rep = {}
        for oc, y in outcomes.items():
            for r in orders:
                res, _ = evaluate_ranking(orders[r], t_ho, y, w)
                stats[(oc, r)].append(res)
                rep[(oc, r)] = res["qini"]
            delta_qini[oc].append(rep[(oc, "x_learner")] - rep[(oc, "response")])
        if (b + 1) % 25 == 0:
            print(f"    [boot] {b+1}/{boot_reps} ({time.time()-t0:.0f}s)")
    cis = {}
    for key, reps in stats.items():
        cis[key] = {
            m: [
                float(np.percentile([r[m] for r in reps], 2.5)),
                float(np.percentile([r[m] for r in reps], 97.5)),
            ]
            for m in reps[0]
        }
    delta_ci = {
        oc: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        for oc, v in delta_qini.items()
    }
    # Paired CATE-minus-response deltas at each k, read off the SAME bootstrap
    # reps already stored above (rep b of x_learner and rep b of response share
    # one resample), so no extra random draws are taken and every number already
    # reported stays bit-for-bit what it was.
    delta_k_ci = {}
    for oc in outcomes:
        reps_x = stats[(oc, "x_learner")]
        reps_r = stats[(oc, "response")]
        per_metric = {}
        for k in UPLIFT_AT_KS:
            m = f"u@{int(k*100)}"
            d = [a[m] - b[m] for a, b in zip(reps_x, reps_r)]
            per_metric[m] = [
                float(np.percentile(d, 2.5)),
                float(np.percentile(d, 97.5)),
            ]
        delta_k_ci[oc] = per_metric
    return points, cis, delta_ci, curves, delta_k_ci


# ---------------------------------------------------------------- hillstrom
def hillstrom_features(df):
    """Deterministic feature matrix: numeric + sorted one-hot dummies."""
    import polars as pl

    num = df.select(
        pl.col("recency").cast(pl.Float32),
        pl.col("history").cast(pl.Float32),
        pl.col("mens").cast(pl.Float32),
        pl.col("womens").cast(pl.Float32),
        pl.col("newbie").cast(pl.Float32),
    )
    dums = df.select(pl.col("zip_code"), pl.col("channel")).to_dummies()
    dums = dums.select(sorted(dums.columns)).cast(pl.Float32)
    feats = pl.concat([num, dums], how="horizontal")
    return feats.to_numpy(), list(feats.columns)


def recency_bucket(r: int) -> str:
    for lo, hi in RECENCY_BUCKETS:
        if lo <= r <= hi:
            return f"{lo}-{hi}m"
    return "10-12m"


def hillstrom_table() -> dict:
    """Womens E-Mail vs No E-Mail. Truth = direct randomized arm contrast per
    segment (NO model). Models (in-sample, illustrative; flagged as such)
    provide the response-rank and predicted-uplift-rank columns."""
    import polars as pl

    df = pl.read_csv(
        HILLSTROM_CSV,
        schema_overrides={"spend": pl.Float64, "history": pl.Float64},
        infer_schema_length=10000,
    )
    df = df.filter(pl.col("segment").is_in(["Womens E-Mail", "No E-Mail"]))
    treat = (df["segment"] == "Womens E-Mail").cast(pl.Int8).to_numpy()
    y_visit = df["visit"].to_numpy().astype(np.int8)
    spend = df["spend"].to_numpy().astype(np.float64)
    X, feat_names = hillstrom_features(df)

    it, ic = np.flatnonzero(treat == 1), np.flatnonzero(treat == 0)
    e = len(it) / (len(it) + len(ic))
    # tiny data -> always single-thread + deterministic so --check retrains identically
    resp, cate = x_learner_scores(
        X[it], y_visit[it], X[ic], y_visit[ic], X, e, threads=1
    )

    rec = df["recency"].to_numpy()
    hseg = df["history_segment"].to_numpy()
    seg_names = np.array(
        [f"{h} x recency {recency_bucket(int(r))}" for h, r in zip(hseg, rec)]
    )
    segments = []
    for name in sorted(set(seg_names.tolist())):
        m = seg_names == name
        mt, mc = m & (treat == 1), m & (treat == 0)
        nt, nc = int(mt.sum()), int(mc.sum())
        vt, vc = float(y_visit[mt].mean()), float(y_visit[mc].mean())
        st, sc = float(spend[mt].mean()), float(spend[mc].mean())
        se_v = float(np.sqrt(vt * (1 - vt) / nt + vc * (1 - vc) / nc))
        se_s = float(np.sqrt(spend[mt].var(ddof=1) / nt + spend[mc].var(ddof=1) / nc))
        segments.append(
            {
                "name": name,
                "n_treated": nt,
                "n_control": nc,
                "model_response_score": float(resp[m].mean()),
                "model_uplift_score": float(cate[m].mean()),
                "measured_visit_uplift_pp": round((vt - vc) * 100, 4),
                "measured_visit_uplift_se_pp": round(se_v * 100, 4),
                "measured_spend_uplift_usd": round(st - sc, 4),
                "measured_spend_uplift_se_usd": round(se_s, 4),
            }
        )
    nseg = len(segments)
    resp_order = np.argsort([-s["model_response_score"] for s in segments], kind="stable")
    upl_order = np.argsort([-s["model_uplift_score"] for s in segments], kind="stable")
    meas_order = np.argsort(
        [-s["measured_visit_uplift_pp"] for s in segments], kind="stable"
    )
    for rank, i in enumerate(resp_order, 1):
        segments[i]["response_rank"] = rank
    for rank, i in enumerate(upl_order, 1):
        segments[i]["uplift_rank"] = rank
    for rank, i in enumerate(meas_order, 1):
        segments[i]["measured_uplift_rank"] = rank
    hi, lo = nseg // 3, nseg - nseg // 3  # terciles
    for s in segments:
        wasted = s["response_rank"] <= hi and (
            s["measured_visit_uplift_pp"] <= 0 or s["measured_uplift_rank"] > lo
        )
        gem = s["response_rank"] > lo and s["measured_uplift_rank"] <= hi
        s["verdict"] = "wasted-budget" if wasted else ("hidden-gem" if gem else "aligned")
    segments.sort(key=lambda s: s["response_rank"])

    overall = {
        "arms": {"womens_email": len(it), "no_email": len(ic)},
        "measured_visit_uplift_pp": round(
            (float(y_visit[it].mean()) - float(y_visit[ic].mean())) * 100, 4
        ),
        "measured_spend_uplift_usd": round(
            float(spend[it].mean()) - float(spend[ic].mean()), 4
        ),
    }
    return {
        "design": "randomized 3-arm e-mail experiment; arms used: Womens E-Mail vs No E-Mail",
        "segmentation": "history_segment x recency buckets (1-3/4-6/7-9/10-12 months)",
        "truth": "per-segment uplift measured DIRECTLY from randomized arms (no model)",
        "model_ranks_note": (
            "response_rank/uplift_rank come from in-sample LightGBM models "
            "(illustrative of how each targeting policy would order segments); "
            "verdicts are decided against the model-free measured uplift"
        ),
        "verdict_rule": (
            "wasted-budget: response rank in top tercile AND measured uplift <=0 "
            "or in bottom tercile; hidden-gem: response rank bottom tercile AND "
            "measured uplift top tercile; else aligned"
        ),
        "overall": overall,
        "n_segments": nseg,
        "segments": segments,
        "features": feat_names,
    }


# ------------------------------------------------------------ orchestration
def criteo_block(points, cis, delta_ci, n_rows, split_info, prop) -> dict:
    def kblock(oc, r):
        return {
            str(int(k * 100)): {
                "value": points[(oc, r)][f"u@{int(k*100)}"],
                "ci": cis[(oc, r)][f"u@{int(k*100)}"],
            }
            for k in UPLIFT_AT_KS
        }

    return {
        "rows": n_rows,
        "split": (
            "pre-registered: stratified-by-treatment 70/30 holdout, seed 42, "
            "committed in scripts/uplift_exhibit.py before the full-data run"
        ),
        "holdout": split_info,
        "propensity": prop,
        "outcome_primary": "conversion",
        "qini_definition": (
            "area between the Qini curve (incremental conversions per customer, "
            "q(phi)/N) and the random-targeting diagonal; not normalized by the "
            "perfect-model area"
        ),
        "qini_x_learner": points[("conversion", "x_learner")]["qini"],
        "qini_response": points[("conversion", "response")]["qini"],
        "qini_ci": {
            "x_learner": cis[("conversion", "x_learner")]["qini"],
            "response": cis[("conversion", "response")]["qini"],
            "delta_x_minus_response": delta_ci["conversion"],
        },
        "uplift_at_k": {
            "x_learner": kblock("conversion", "x_learner"),
            "response": kblock("conversion", "response"),
        },
        "visit_robustness": {
            "qini_x_learner": points[("visit", "x_learner")]["qini"],
            "qini_response": points[("visit", "response")]["qini"],
            "qini_ci": {
                "x_learner": cis[("visit", "x_learner")]["qini"],
                "response": cis[("visit", "response")]["qini"],
                "delta_x_minus_response": delta_ci["visit"],
            },
            "uplift_at_k": {
                "x_learner": kblock("visit", "x_learner"),
                "response": kblock("visit", "response"),
            },
        },
        "ci_method": "stratified bootstrap",
        "ci_detail": (
            f"within-treatment-arm resampling with replacement, B={BOOTSTRAP_REPS}, "
            "percentile 95% CIs; the same resample evaluates both rankings "
            "(paired), so delta CIs are paired"
        ),
        "bootstrap_reps": BOOTSTRAP_REPS,
    }


TARGETING_OUTCOMES = ("visit", "conversion")


def targeting_block(points, cis, delta_k_ci, rates, boot_reps: int) -> dict:
    """The `criteo.targeting_at_k` block (CONTRACT §2).

    For each outcome and each k, four numbers with the same stratified-bootstrap
    CI used everywhere else in this script: the incremental outcome rate inside
    the targeted top-k under CATE ranking, the same under response ranking, the
    paired difference between them, and the whole-holdout ATE as the reference
    line. Reported as obtained, including the k values where CATE ranking loses.
    """
    out = {
        "definition": (
            "incremental outcome rate inside the targeted top-k of the "
            "randomized holdout, measured as treated-arm rate minus control-arm "
            "rate among the rows a ranking would target; ate = the same "
            "difference over the whole holdout (ranking-independent)"
        ),
        "ranking_names": {
            "cate": "X-learner predicted CATE (uplift ranking)",
            "response": "LightGBM P(outcome|X) fitted on the treated arm "
                        "(response-propensity ranking)",
        },
        "ci_method": "stratified bootstrap",
        "ci_detail": (
            f"within-treatment-arm resampling with replacement, B={boot_reps}, "
            "percentile 95% CIs; one resample per rep evaluates both rankings "
            "and both outcomes, so the difference CIs are paired"
        ),
        "bootstrap_reps": boot_reps,
        "outcomes": {},
    }
    for oc in TARGETING_OUTCOMES:
        ate_x = points[(oc, "x_learner")]["ate"]
        ate_r = points[(oc, "response")]["ate"]
        if abs(ate_x - ate_r) > 1e-12:
            raise SystemExit(f"ATE is not ranking-independent for {oc}: {ate_x} vs {ate_r}")
        blk = {
            "treated_rate": rates[oc]["treated_rate"],
            "control_rate": rates[oc]["control_rate"],
            "ate": {"value": ate_x, "ci": cis[(oc, "x_learner")]["ate"]},
            "k": {},
        }
        for k in UPLIFT_AT_KS:
            kk = str(int(k * 100))
            m = f"u@{kk}"
            v_c = points[(oc, "x_learner")][m]
            v_r = points[(oc, "response")][m]
            blk["k"][kk] = {
                "cate_ranking": {"value": v_c, "ci": cis[(oc, "x_learner")][m]},
                "response_ranking": {"value": v_r, "ci": cis[(oc, "response")][m]},
                "difference_cate_minus_response": {
                    "value": v_c - v_r,
                    "ci": delta_k_ci[oc][m],
                },
                "cate_over_ate": (v_c / ate_x) if ate_x else None,
            }
        out["outcomes"][oc] = blk
    return out


def holdout_rates(t_ho, outcomes: dict) -> dict:
    rates = {}
    for oc, y in outcomes.items():
        rt = float(y[t_ho == 1].mean())
        rc = float(y[t_ho == 0].mean())
        rates[oc] = {
            "treated_rate": rt,
            "control_rate": rc,
            "ate_pp": round((rt - rc) * 100, 5),
        }
    return rates


def evaluate_from_cache(cache, boot_reps: int):
    """Merged points/cis/delta_ci/delta_k_ci over both outcomes, from the
    committed prediction cache. Same call shape run() uses."""
    t_ho = cache["t_ho"]
    points, cis, delta_ci, curves, delta_k_ci = {}, {}, {}, {}, {}
    for oc in ("conversion", "visit"):
        p, c, d, cv, dk = holdout_evaluation(
            t_ho,
            {oc: cache[f"y_{oc}"]},
            {"response": cache[f"resp_{oc}"], "x_learner": cache[f"cate_{oc}"]},
            boot_reps,
        )
        points.update(p)
        cis.update(c)
        delta_ci.update(d)
        curves.update(cv)
        delta_k_ci.update(dk)
    return points, cis, delta_ci, curves, delta_k_ci


def augment_targeting(dev: bool) -> int:
    """Compute `criteo.targeting_at_k` from the committed cache and merge it
    into an existing results/uplift.json. Every pre-existing key is asserted
    byte-identical before the file is rewritten."""
    out_dir = RESULTS / "dev" if dev else RESULTS
    json_path = out_dir / "uplift.json"
    raw = json_path.read_bytes()
    stored = json.loads(raw)
    if json.dumps(stored, indent=1).encode() != raw:
        return _fail("uplift.json does not round-trip; refusing to rewrite it")
    cache = np.load(PROJ / stored["check"]["cache"])
    boot_reps = int(cache["boot_reps"])
    print(f"[targeting] evaluating from cache, B={boot_reps}")
    points, cis, _, _, delta_k_ci = evaluate_from_cache(cache, boot_reps)
    rates = holdout_rates(
        cache["t_ho"], {"conversion": cache["y_conversion"], "visit": cache["y_visit"]}
    )
    stored["criteo"]["targeting_at_k"] = targeting_block(
        points, cis, delta_k_ci, rates, boot_reps
    )
    new_raw = json.dumps(stored, indent=1).encode()
    probe = json.loads(new_raw)
    probe["criteo"].pop("targeting_at_k")
    if json.dumps(probe, indent=1).encode() != raw:
        return _fail("merge would change a pre-existing key; nothing written")
    json_path.write_bytes(new_raw)
    print(f"[targeting] wrote criteo.targeting_at_k into {json_path} "
          f"({len(raw)} -> {len(new_raw)} bytes, existing keys byte-identical)")
    for oc in TARGETING_OUTCOMES:
        b = stored["criteo"]["targeting_at_k"]["outcomes"][oc]
        print(f"  {oc}: ate={b['ate']['value']:.6g}")
        for kk, e in b["k"].items():
            print(f"    k={kk}%  cate={e['cate_ranking']['value']:.6g} "
                  f"response={e['response_ranking']['value']:.6g} "
                  f"diff={e['difference_cate_minus_response']['value']:.6g} "
                  f"ci={[round(v, 8) for v in e['difference_cate_minus_response']['ci']]}")
    return 0


def _fail(msg: str) -> int:
    print(f"TARGETING FAIL: {msg}")
    return 1


def run(dev: bool) -> None:
    import lightgbm
    import polars
    import pyarrow
    import sklearn

    t_start = time.time()
    out_dir = RESULTS / "dev" if dev else RESULTS
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    boot_reps = 50 if dev else BOOTSTRAP_REPS

    ensure_parquet()
    print(f"[load] criteo parquet (dev={dev})")
    X, t, yc, yv, n_rows = load_criteo(dev)
    print(f"[load] n={n_rows:,} treated_share={float(t.mean()):.4f}")

    rng = np.random.default_rng(SEED)
    idx_t = np.flatnonzero(t == 1)
    idx_c = np.flatnonzero(t == 0)
    perm_t = rng.permutation(idx_t)
    perm_c = rng.permutation(idx_c)
    cut_t = int(round(len(perm_t) * (1 - HOLDOUT_FRAC)))
    cut_c = int(round(len(perm_c) * (1 - HOLDOUT_FRAC)))
    tr_idx = np.sort(np.concatenate([perm_t[:cut_t], perm_c[:cut_c]]))
    ho_idx = np.sort(np.concatenate([perm_t[cut_t:], perm_c[cut_c:]]))

    tt = tr_idx[t[tr_idx] == 1]
    ct = tr_idx[t[tr_idx] == 0]
    X_tt, y_tt = X[tt], {"conversion": yc[tt], "visit": yv[tt]}
    X_ct, y_ct = X[ct], {"conversion": yc[ct], "visit": yv[ct]}
    X_ho = X[ho_idx]
    t_ho = t[ho_idx]
    outcomes_ho = {"conversion": yc[ho_idx], "visit": yv[ho_idx]}
    e_train = float(len(tt) / (len(tt) + len(ct)))
    del X

    split_info = {
        "n_train": int(len(tr_idx)),
        "n_holdout": int(len(ho_idx)),
        "n_holdout_treated": int((t_ho == 1).sum()),
        "n_holdout_control": int((t_ho == 0).sum()),
        "holdout_treated_share": float(t_ho.mean()),
    }
    print(f"[split] {split_info}")

    scores = {}
    for oc in ("conversion", "visit"):
        print(f"[train] outcome={oc}")
        resp, cate = x_learner_scores(
            X_tt, y_tt[oc], X_ct, y_ct[oc], X_ho, e_train, TRAIN_THREADS
        )
        scores[oc] = {"response": resp, "x_learner": cate}
    del X_tt, X_ct, X_ho

    cache_path = cache_dir / "uplift_scores.npz"
    np.savez_compressed(
        cache_path,
        t_ho=t_ho,
        y_conversion=outcomes_ho["conversion"],
        y_visit=outcomes_ho["visit"],
        resp_conversion=scores["conversion"]["response"],
        cate_conversion=scores["conversion"]["x_learner"],
        resp_visit=scores["visit"]["response"],
        cate_visit=scores["visit"]["x_learner"],
        boot_reps=np.int64(boot_reps),
        n_rows=np.int64(n_rows),
    )
    print(f"[cache] {cache_path} ({cache_path.stat().st_size/1e6:.0f} MB)")

    print(f"[eval] point estimates + stratified bootstrap B={boot_reps}")
    points, cis, delta_ci, curves, delta_k_ci = {}, {}, {}, {}, {}
    for oc in ("conversion", "visit"):
        p, c, d, cv, dk = holdout_evaluation(
            t_ho,
            {oc: outcomes_ho[oc]},
            {
                "response": scores[oc]["response"],
                "x_learner": scores[oc]["x_learner"],
            },
            boot_reps,
        )
        points.update(p)
        cis.update(c)
        delta_ci.update(d)
        curves.update(cv)
        delta_k_ci.update(dk)

    rates = holdout_rates(t_ho, outcomes_ho)

    prop = {
        "design": (
            "randomized; treatment propensity is a known constant ~0.85 by "
            "construction of Criteo Uplift v2.1 (85/15 random assignment)"
        ),
        "measured_train_share": e_train,
        "used_in_x_learner_blend": e_train,
    }

    print("[hillstrom] segment table")
    hill = hillstrom_table()

    shas = read_checksums()
    result = {
        "seed": SEED,
        "versions": {
            "python": sys.version.split()[0],
            "polars": polars.__version__,
            "numpy": np.__version__,
            "lightgbm": lightgbm.__version__,
            "scikit-learn": sklearn.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "generated_by": "scripts/uplift_exhibit.py --check-able",
        "data_sources": [
            {
                "name": "Criteo Uplift v2.1 (randomized 85/15)",
                "url": CRITEO_URL,
                "sha256": shas.get("criteo-uplift-v2.1.csv.gz", "MISSING"),
            },
            {
                "name": "Hillstrom MineThatData E-Mail (randomized 3-arm)",
                "url": HILLSTROM_URL,
                "sha256": shas.get("hillstrom.csv", "MISSING"),
            },
        ],
        "labels": ["proxy", "randomized-experiment"],
        "dev_mode": dev,
        "criteo": {
            **criteo_block(points, cis, delta_ci, n_rows, split_info, prop),
            "holdout_rates": rates,
            "targeting_at_k": targeting_block(
                points, cis, delta_k_ci, rates, boot_reps
            ),
        },
        "hillstrom": hill,
        "dunnhumby": {
            "included": False,
            "assignment": "observational-targeted",
            "estimator": "doubly-robust",
            "overlap_diagnostic": None,
            "note": (
                "dataset not fetched Day 1 (zip URL absent from the "
                "dunnhumby source-files page); by design the observational "
                "side panel is omitted rather than shown without overlap "
                "diagnostics"
            ),
        },
        "figures": {
            "qini_conversion": {
                "phi": curves[("conversion", "x_learner")][0],
                "x_learner": curves[("conversion", "x_learner")][1],
                "response": curves[("conversion", "response")][1],
                "units": "incremental conversions per customer (q(phi)/N)",
            },
            "qini_visit": {
                "phi": curves[("visit", "x_learner")][0],
                "x_learner": curves[("visit", "x_learner")][1],
                "response": curves[("visit", "response")][1],
                "units": "incremental visits per customer (q(phi)/N)",
            },
        },
        "check": {
            "cache": str(cache_path.relative_to(PROJ)),
            "tolerance": TOL,
            "procedure": (
                "scripts/uplift_exhibit.py --check re-evaluates the holdout "
                "metrics + bootstrap from the saved prediction cache and "
                "retrains the single-thread deterministic Hillstrom models; "
                "asserts |a-b| <= 1e-6 on every reported number"
            ),
        },
        "runtime": {
            "train_threads": TRAIN_THREADS,
            "wall_seconds": round(time.time() - t_start, 1),
        },
    }
    out_path = out_dir / "uplift.json"
    out_path.write_text(json.dumps(result, indent=1))
    print(f"[done] {out_path} ({out_path.stat().st_size/1e3:.0f} KB) "
          f"in {result['runtime']['wall_seconds']}s")
    print(
        f"[headline] conversion qini: x_learner={result['criteo']['qini_x_learner']:.6g} "
        f"response={result['criteo']['qini_response']:.6g} "
        f"delta_ci={result['criteo']['qini_ci']['delta_x_minus_response']}"
    )


# -------------------------------------------------------------------- check
def approx(a, b) -> bool:
    return abs(float(a) - float(b)) <= TOL


def check(dev: bool) -> int:
    out_dir = RESULTS / "dev" if dev else RESULTS
    json_path = out_dir / "uplift.json"
    stored = json.loads(json_path.read_text())
    cache = np.load(PROJ / stored["check"]["cache"])
    boot_reps = int(cache["boot_reps"])
    t_ho = cache["t_ho"]
    failures = []

    def cmp(name, a, b):
        if not approx(a, b):
            failures.append(f"{name}: recomputed {a!r} != stored {b!r}")

    all_points, all_cis, all_delta_k = {}, {}, {}
    for oc in ("conversion", "visit"):
        points, cis, delta_ci, curves, delta_k_ci = holdout_evaluation(
            t_ho,
            {oc: cache[f"y_{oc}"]},
            {"response": cache[f"resp_{oc}"], "x_learner": cache[f"cate_{oc}"]},
            boot_reps,
        )
        all_points.update(points)
        all_cis.update(cis)
        all_delta_k.update(delta_k_ci)
        blk = stored["criteo"] if oc == "conversion" else stored["criteo"]["visit_robustness"]
        cmp(f"{oc}.qini_x_learner", points[(oc, "x_learner")]["qini"], blk["qini_x_learner"])
        cmp(f"{oc}.qini_response", points[(oc, "response")]["qini"], blk["qini_response"])
        for r in ("x_learner", "response"):
            for lohi in (0, 1):
                cmp(
                    f"{oc}.qini_ci.{r}[{lohi}]",
                    cis[(oc, r)]["qini"][lohi],
                    blk["qini_ci"][r][lohi],
                )
            for k in UPLIFT_AT_KS:
                kk = str(int(k * 100))
                cmp(
                    f"{oc}.u@{kk}.{r}",
                    points[(oc, r)][f"u@{kk}"],
                    blk["uplift_at_k"][r][kk]["value"],
                )
                for lohi in (0, 1):
                    cmp(
                        f"{oc}.u@{kk}.{r}.ci[{lohi}]",
                        cis[(oc, r)][f"u@{kk}"][lohi],
                        blk["uplift_at_k"][r][kk]["ci"][lohi],
                    )
        for lohi in (0, 1):
            cmp(
                f"{oc}.delta_ci[{lohi}]",
                delta_ci[oc][lohi],
                blk["qini_ci"]["delta_x_minus_response"][lohi],
            )
        fig = stored["figures"][f"qini_{oc}"]
        for cname, r in (("x_learner", "x_learner"), ("response", "response")):
            rec = curves[(oc, r)][1]
            for i, (a, b) in enumerate(zip(rec, fig[cname])):
                if not approx(a, b):
                    failures.append(f"figures.qini_{oc}.{cname}[{i}]: {a} != {b}")
                    break

    tgt = stored["criteo"].get("targeting_at_k")
    if tgt is None:
        failures.append("criteo.targeting_at_k missing (run --targeting)")
    else:
        rates = holdout_rates(
            t_ho, {"conversion": cache["y_conversion"], "visit": cache["y_visit"]}
        )
        rebuilt = targeting_block(
            all_points, all_cis, all_delta_k, rates, int(tgt["bootstrap_reps"])
        )
        for oc in TARGETING_OUTCOMES:
            a, b = rebuilt["outcomes"][oc], tgt["outcomes"][oc]
            cmp(f"targeting.{oc}.treated_rate", a["treated_rate"], b["treated_rate"])
            cmp(f"targeting.{oc}.control_rate", a["control_rate"], b["control_rate"])
            cmp(f"targeting.{oc}.ate", a["ate"]["value"], b["ate"]["value"])
            for lohi in (0, 1):
                cmp(f"targeting.{oc}.ate.ci[{lohi}]",
                    a["ate"]["ci"][lohi], b["ate"]["ci"][lohi])
            for kk in a["k"]:
                for field in ("cate_ranking", "response_ranking",
                              "difference_cate_minus_response"):
                    cmp(f"targeting.{oc}.{kk}.{field}",
                        a["k"][kk][field]["value"], b["k"][kk][field]["value"])
                    for lohi in (0, 1):
                        cmp(f"targeting.{oc}.{kk}.{field}.ci[{lohi}]",
                            a["k"][kk][field]["ci"][lohi],
                            b["k"][kk][field]["ci"][lohi])
                cmp(f"targeting.{oc}.{kk}.cate_over_ate",
                    a["k"][kk]["cate_over_ate"], b["k"][kk]["cate_over_ate"])

    hill = hillstrom_table()
    for s_new, s_old in zip(hill["segments"], stored["hillstrom"]["segments"]):
        if s_new["name"] != s_old["name"]:
            failures.append(f"hillstrom segment order: {s_new['name']} != {s_old['name']}")
            break
        for key in (
            "response_rank",
            "uplift_rank",
            "measured_uplift_rank",
            "measured_visit_uplift_pp",
            "measured_spend_uplift_usd",
            "model_response_score",
            "model_uplift_score",
        ):
            cmp(f"hillstrom.{s_new['name']}.{key}", s_new[key], s_old[key])
        if s_new["verdict"] != s_old["verdict"]:
            failures.append(f"hillstrom.{s_new['name']}.verdict differs")

    if failures:
        print(f"CHECK FAIL ({len(failures)} mismatches at tol {TOL}):")
        for f in failures[:20]:
            print("  -", f)
        return 1
    print(f"CHECK PASS: all reported numbers reproduce within {TOL} "
          f"(criteo from cache, hillstrom retrained single-thread)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", action="store_true", help="1M-row dev loop -> results/dev/")
    ap.add_argument("--check", action="store_true", help="verify from cache; exit 0/1")
    ap.add_argument(
        "--targeting",
        action="store_true",
        help="compute criteo.targeting_at_k from the committed cache and merge "
             "it into an existing uplift.json (pre-existing keys untouched)",
    )
    args = ap.parse_args()
    if args.check:
        return check(args.dev)
    if args.targeting:
        return augment_targeting(args.dev)
    run(args.dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
