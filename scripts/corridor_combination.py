#!/usr/bin/env python3
"""Corridor forecast combination (WS-B2 follow-up).

Pre-registered in CORRIDOR-COMBINATION-PREREG.md, committed alone at commit
418aef3 before this producer was written and before any combination number was
computed. Nothing here was tuned: the weights are fixed at 0.5 naive plus 0.5
model, no other weight and no other combination is computed, and the decision
sentence is written by this script from the numbers.

Question: the corridor exhibit reports the global LightGBM losing pooled
accuracy to the per-corridor seasonal naive (macro MASE 0.622977 against
0.530169 over the 13-month holdout). The deployment question is not whether the
model replaces the naive but whether adding it helps. Bates and Granger (1969):
unweighted averages of imperfectly correlated forecasts often beat both parents.

Method:
  * The shipped pipeline's one-month-ahead rolling holdout forecasts (2025-01 to
    2026-01, 13 months, 12 corridors) are recomputed deterministically by
    importing scripts/corridor_exhibit.py (not edited, not copied). Only the
    train-predict-and-score block is reimplemented here, because
    corridor_exhibit.compute() returns the finished JSON and never exposes the
    forecast matrices; each reimplemented step names the shipped lines it
    mirrors.
  * HARD GATE before any combination math: the recomputed per-corridor
    mase_model and mase_seasonal_naive, and the two macro means, must reproduce
    results/corridor.json at 1e-6. On failure the run aborts and nothing is
    written.
  * Combination: 0.5 * naive + 0.5 * model, per corridor per month, scored on
    the identical MASE scale (in-sample mean |y_t - y_{t-12}| over the train
    window), macro-averaged over the 12 corridors.
  * No interval is claimed. 13 months and 12 corridors: point comparison, the
    exhibit's own convention.

Usage:
  ./.venv/bin/python scripts/corridor_combination.py           # compute + write
  ./.venv/bin/python scripts/corridor_combination.py --check   # reproduce at 1e-6
"""

from __future__ import annotations

import os

# Cap threads BEFORE the numeric imports (8GB shared machine; determinism).
# corridor_exhibit itself pins the model to n_jobs=1 and deterministic=True.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import corridor_exhibit as ce

REPO = Path(__file__).resolve().parent.parent
OUT_PATH = REPO / "results" / "corridor_combination.json"
SHIPPED_PATH = REPO / "results" / "corridor.json"

SEED = ce.SEED
TOL = 1e-6

PREREG_FILE = "CORRIDOR-COMBINATION-PREREG.md"
PREREG_COMMIT = "418aef3"

# Fixed a priori by the pre-registration. Not fitted, not searched.
W_NAIVE = 0.5
W_MODEL = 0.5

# The shipped reference numbers this analysis is measured against, restated from
# results/corridor.json and re-verified by the reproduction gate below.
SHIPPED_MACRO_NAIVE = 0.530169
SHIPPED_MACRO_MODEL = 0.622977

CAVEATS = [
    "13 holdout months and 12 corridors: this is a point comparison and no "
    "interval is claimed for it, which is the corridor exhibit's own convention",
    "the weights are fixed a priori at 0.5 and 0.5 by the pre-registration; no "
    "weight was fitted, searched or reported, so no in-sample weight-selection "
    "advantage enters this number",
    "the target is SingStat international visitor arrivals by place of "
    "residence, a stated proxy for cross-border card-spend corridors, so the "
    "combination is validated on the proxy and not on card spend",
    "the MASE scale is the shipped in-sample mean |y_t - y_{t-12}| over "
    "2009-01..2024-12, which spans the COVID break; it is identical for the "
    "naive, the model and the combination, so the comparison between them is "
    "unaffected",
    "the combination reuses the shipped base forecasts, not the MinT-reconciled "
    "ones; reconciliation is a separate axis reported in results/corridor.json "
    "and is untouched here",
    "no other combination rule was computed: this file holds one weighting and "
    "the numbers it produced on the first and only run",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recompute_shipped_forecasts():
    """Recompute the shipped pipeline's holdout forecasts deterministically.

    Everything upstream of the fit is imported from corridor_exhibit: checksum
    verification, the SingStat parse, corridor selection, the KNOMAD row and the
    feature table. The block below mirrors corridor_exhibit.compute() lines
    321-369 and exists only because compute() builds the finished JSON in one
    body and returns no intermediates.
    """
    checksums = ce.verify_checksums()                              # ce line 315
    months, countries = ce.parse_singstat()                        # ce line 316
    node_names, mat, top, region_of, regions, T = ce.build_series(  # ce line 317
        months, countries)
    knomad_sg = ce.load_knomad_sg_row()                            # ce line 318
    X, yv, keys = ce.make_features(node_names, mat, months, knomad_sg, top)

    # mirrors ce lines 321-325
    is_train = np.array([ce.month_leq(ce.FEATURE_START, m) and ce.month_leq(m, ce.TRAIN_END)
                         for (_i, m) in keys])
    is_hold = np.array([ce.month_leq(ce.HOLDOUT_START, m) and ce.month_leq(m, ce.HOLDOUT_END)
                        for (_i, m) in keys])

    import lightgbm as lgb

    # mirrors ce lines 329-335, identical hyperparameters and seed
    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, subsample=1.0,
        colsample_bytree=1.0, random_state=SEED, deterministic=True,
        force_row_wise=True, n_jobs=1, verbose=-1)
    cat_idx = [ce.FEATURES.index(c) for c in ce.CATEGORICAL]
    model.fit(X[is_train], yv[is_train], categorical_feature=cat_idx)

    # mirrors ce lines 337-339
    pred = np.clip(np.expm1(model.predict(X)), 0.0, None)
    actual = np.expm1(yv)

    # mirrors ce lines 341-351
    n_nodes = len(node_names)
    hold_months = sorted({m for ((_i, m), hld) in zip(keys, is_hold) if hld})
    h = len(hold_months)
    m2col = {m: j for j, m in enumerate(hold_months)}
    base_fc = np.full((n_nodes, h), np.nan)
    actual_h = np.full((n_nodes, h), np.nan)
    for r, ((i, m), hld) in enumerate(zip(keys, is_hold)):
        if hld:
            base_fc[i, m2col[m]] = pred[r]
            actual_h[i, m2col[m]] = actual[r]

    # mirrors ce lines 354-363: seasonal naive on the holdout, in-sample scale
    month_index = {m: t for t, m in enumerate(months)}
    naive_fc = np.full((n_nodes, h), np.nan)
    scale = np.full(n_nodes, np.nan)
    for i in range(n_nodes):
        y = mat[i]
        for m, j in m2col.items():
            naive_fc[i, j] = y[month_index[m] - 12]
        tr = [t for t, m in enumerate(months)
              if ce.month_leq(ce.FEATURE_START, m) and ce.month_leq(m, ce.TRAIN_END)]
        scale[i] = np.mean([abs(y[t] - y[t - 12]) for t in tr])

    return {
        "checksums": checksums,
        "node_names": node_names,
        "top": top,
        "region_of": region_of,
        "hold_months": hold_months,
        "base_fc": base_fc,
        "naive_fc": naive_fc,
        "actual_h": actual_h,
        "scale": scale,
    }


def make_scorers(actual_h, scale):
    """mirrors ce lines 365-369: MAE and MASE per node on the holdout."""
    def mae(fc, i):
        return float(np.mean(np.abs(fc[i] - actual_h[i])))

    def mase(fc, i):
        return float(mae(fc, i) / scale[i])

    return mae, mase


def reproduction_gate(fc, mase, shipped):
    """HARD GATE. The recomputed per-corridor and macro MASE must reproduce the
    shipped results/corridor.json at 1e-6. Nothing downstream runs otherwise."""
    node_names, top = fc["node_names"], fc["top"]
    shipped_by_origin = {c["origin"]: c for c in shipped["corridors"]}

    missing = [c for c in top if c not in shipped_by_origin]
    extra = [c for c in shipped_by_origin if c not in top]
    if missing or extra:
        raise SystemExit(
            "REPRODUCTION GATE FAILED: corridor set does not match "
            f"results/corridor.json. recomputed-only={missing} shipped-only={extra}")

    rows, diffs_model, diffs_naive = [], [], []
    for c in top:
        i = node_names.index(c)
        got_model = mase(fc["base_fc"], i)
        got_naive = mase(fc["naive_fc"], i)
        want_model = float(shipped_by_origin[c]["mase_model"])
        want_naive = float(shipped_by_origin[c]["mase_seasonal_naive"])
        d_model = abs(got_model - want_model)
        d_naive = abs(got_naive - want_naive)
        diffs_model.append(d_model)
        diffs_naive.append(d_naive)
        rows.append({"origin": c, "abs_diff_mase_model": d_model,
                     "abs_diff_mase_seasonal_naive": d_naive})

    bottom_idx = [node_names.index(c) for c in top]
    macro_model = float(np.mean([mase(fc["base_fc"], i) for i in bottom_idx]))
    macro_naive = float(np.mean([mase(fc["naive_fc"], i) for i in bottom_idx]))
    d_macro_model = abs(macro_model - float(shipped["mase_model"]))
    d_macro_naive = abs(macro_naive - float(shipped["mase_seasonal_naive"]))

    max_abs_diff = max(max(diffs_model), max(diffs_naive),
                       d_macro_model, d_macro_naive)
    ok = max_abs_diff <= TOL

    gate = {
        "what": "recomputed per-corridor and macro MASE against the committed "
                "results/corridor.json, before any combination math",
        "n_corridors_compared": len(top),
        "n_values_compared": 2 * len(top) + 2,
        "max_abs_diff_mase_model": round(max(diffs_model), 12),
        "max_abs_diff_mase_seasonal_naive": round(max(diffs_naive), 12),
        "max_abs_diff_macro_model": round(d_macro_model, 12),
        "max_abs_diff_macro_seasonal_naive": round(d_macro_naive, 12),
        "max_abs_diff": round(max_abs_diff, 12),
        "tolerance": TOL,
        "ok": ok,
        "note": "the committed values are stored rounded to 6 decimals, so a "
                "perfect recomputation still shows a diff up to 5e-7",
    }
    if not ok:
        worst = max(rows, key=lambda r: max(r["abs_diff_mase_model"],
                                            r["abs_diff_mase_seasonal_naive"]))
        raise SystemExit(
            "REPRODUCTION GATE FAILED: recomputed corridor MASE does not match "
            f"results/corridor.json at {TOL}.\n"
            f"  max abs diff = {max_abs_diff:.3e}\n"
            f"  worst corridor = {worst['origin']} "
            f"(model {worst['abs_diff_mase_model']:.3e}, "
            f"naive {worst['abs_diff_mase_seasonal_naive']:.3e})\n"
            f"  macro model diff = {d_macro_model:.3e}, "
            f"macro naive diff = {d_macro_naive:.3e}\n"
            "ABORTING before the combination is evaluated. Nothing written.")
    return gate, macro_model, macro_naive


def write_sentence(comb_macro, naive_macro, model_macro, wins, n_corridors,
                   n_months, first_month, last_month):
    """The pre-registered decision rule, written from the numbers by the script.

    Prereg: combination macro MASE < naive macro MASE means the model adds value
    as a complement and the margin is stated; otherwise the finding is that it
    does not help even as a complement, shipped at the same prominence.
    """
    margin = naive_macro - comb_macro          # positive means the combination wins
    pct = 100.0 * margin / naive_macro
    common = (
        f"The weights were fixed at {W_NAIVE} and {W_MODEL} before the run in "
        f"{PREREG_FILE} (commit {PREREG_COMMIT}), no other weight was computed, "
        f"and with {n_months} holdout months across {n_corridors} corridors this "
        "is a point comparison with no interval claimed."
    )
    if comb_macro < naive_macro:
        branch = "combination_beats_naive"
        sentence = (
            f"Averaging the model with the seasonal naive at fixed equal weights "
            f"beats the naive on the same {first_month} to {last_month} holdout: "
            f"macro MASE {comb_macro:.4f} against the naive's {naive_macro:.4f}, "
            f"a margin of {margin:.4f} ({pct:.1f} percent lower error), with the "
            f"model alone at {model_macro:.4f}. The model does not replace the "
            f"cheap baseline, but adding it to the baseline at equal weights "
            f"improves the baseline, and the combination beats the naive in "
            f"{wins} of the {n_corridors} corridors. {common}"
        )
    else:
        branch = "combination_does_not_beat_naive"
        gap = comb_macro - naive_macro
        sentence = (
            f"Averaging the model with the seasonal naive at fixed equal weights "
            f"does not beat the naive on the same {first_month} to {last_month} "
            f"holdout: macro MASE {comb_macro:.4f} against the naive's "
            f"{naive_macro:.4f}, {gap:.4f} worse, with the model alone at "
            f"{model_macro:.4f}. The model loses to the cheap baseline on its own "
            f"and it also fails to help as a complement at equal weights, beating "
            f"the naive in only {wins} of the {n_corridors} corridors, and that is "
            f"the finding. {common}"
        )
    return branch, sentence, margin, pct


def build():
    fc = recompute_shipped_forecasts()
    mae, mase = make_scorers(fc["actual_h"], fc["scale"])

    if not SHIPPED_PATH.is_file():
        raise SystemExit(f"REPRODUCTION GATE FAILED: {SHIPPED_PATH} does not exist")
    shipped = json.loads(SHIPPED_PATH.read_text())

    # --- HARD GATE: nothing below runs unless the shipped numbers reproduce ---
    gate, macro_model, macro_naive = reproduction_gate(fc, mase, shipped)

    # --- the pre-registered combination, the only one computed ---------------
    comb_fc = W_NAIVE * fc["naive_fc"] + W_MODEL * fc["base_fc"]

    node_names, top = fc["node_names"], fc["top"]
    bottom_idx = [node_names.index(c) for c in top]
    macro_comb = float(np.mean([mase(comb_fc, i) for i in bottom_idx]))

    corridors, wins = [], 0
    for c in top:
        i = node_names.index(c)
        m_model = mase(fc["base_fc"], i)
        m_naive = mase(fc["naive_fc"], i)
        m_comb = mase(comb_fc, i)
        beats = bool(m_comb < m_naive)
        wins += int(beats)
        corridors.append({
            "origin": c,
            "region": fc["region_of"][c],
            "mase_seasonal_naive": round(m_naive, 6),
            "mase_model": round(m_model, 6),
            "mase_combination": round(m_comb, 6),
            # from the reported 6-decimal values, so every delta in this file
            # is the difference a reader can recompute from the columns
            "combination_minus_naive": round(round(m_comb, 6) - round(m_naive, 6), 6),
            "combination_beats_naive": beats,
            "mae_seasonal_naive": round(mae(fc["naive_fc"], i), 2),
            "mae_model": round(mae(fc["base_fc"], i), 2),
            "mae_combination": round(mae(comb_fc, i), 2),
        })

    hold_months = fc["hold_months"]
    n_months = len(hold_months)

    def label(m):
        return f"{m[0]:04d}-{m[1]:02d}"

    # the reported (6-decimal) values are the ones the decision rule reads, so
    # the JSON and the sentence can never disagree
    r_comb = round(macro_comb, 6)
    r_naive = round(macro_naive, 6)
    r_model = round(macro_model, 6)
    branch, sentence, margin, pct = write_sentence(
        r_comb, r_naive, r_model, wins, len(top), n_months,
        label(hold_months[0]), label(hold_months[-1]))

    import lightgbm

    return {
        "seed": SEED,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "generated_by": "scripts/corridor_combination.py --check-able",
        "data_sources": [
            {"name": "SingStat International Visitor Arrivals by place of "
                     "residence, monthly (data.gov.sg, Open Data Licence)",
             "url": "https://data.gov.sg/datasets/d_7e7b2ee60c6ffc962f80fef129cf306e/view",
             "sha256": fc["checksums"]["singstat_iva_monthly.csv"]},
            {"name": "KNOMAD bilateral remittance matrix 2021 (static gravity "
                     "covariates only)",
             "url": "https://web.archive.org/web/20230424090916/https://knomad.org/sites/default/files/2022-12/bilateral_remittance_matrix_2021_0.xlsx",
             "sha256": fc["checksums"]["knomad_bilateral_2021.xlsx"]},
        ],
        "labels": [
            "real",
            "proxy: tourism arrivals stand in for cross-border card spend",
        ],
        "what_this_is": (
            "the pre-registered test of whether ADDING the corridor model to the "
            "seasonal naive helps, at weights fixed before the run; the shipped "
            "exhibit already reports the model losing to the naive alone and that "
            "loss stays printed"),
        "pre_registration": {
            "file": PREREG_FILE,
            "commit": PREREG_COMMIT,
            "committed_before_any_combination_number_was_computed": True,
            "weights": {"seasonal_naive": W_NAIVE, "model": W_MODEL,
                        "fixed_a_priori": True,
                        "note": "the equal-weights convention (Bates and Granger "
                                "1969); no other weight computed, reported or tried"},
            "decision_rule": {
                "combination_beats_naive": "combination macro MASE < naive macro "
                                           "MASE: the model adds value as a "
                                           "complement, margin stated",
                "combination_does_not_beat_naive": "combination macro MASE >= naive "
                                                   "macro MASE: the model does not "
                                                   "add value even as a complement "
                                                   "at equal weights, shipped at the "
                                                   "same prominence",
            },
            "sentence_authorship": "written by this script from the numbers, "
                                   "never by hand",
        },
        "shipped_reference": {
            "file": "results/corridor.json",
            "sha256": sha256_of(SHIPPED_PATH),
            "producer": "scripts/corridor_exhibit.py (imported here, not edited)",
            "mase_seasonal_naive": SHIPPED_MACRO_NAIVE,
            "mase_model": SHIPPED_MACRO_MODEL,
            "mase_definition": "macro mean of per-corridor MASE over the 12 corridors",
        },
        "design": {
            "forecasts": "the shipped one-month-ahead rolling holdout forecasts, "
                         "recomputed deterministically by importing "
                         "scripts/corridor_exhibit.py (verify_checksums, "
                         "parse_singstat, build_series, load_knomad_sg_row, "
                         "make_features, month_leq and its constants); only the "
                         "fit-predict-and-score block is reimplemented, because "
                         "compute() returns the finished JSON and exposes no "
                         "forecast matrices",
            "combination": f"{W_NAIVE} * seasonal naive + {W_MODEL} * model, per "
                           "corridor per month",
            "holdout": f"{label(hold_months[0])}..{label(hold_months[-1])} "
                       f"({n_months} months, {len(top)} corridors)",
            "mase_scale": "the shipped in-sample mean |y_t - y_{t-12}| over "
                          "2009-01..2024-12, identical for naive, model and "
                          "combination",
            "interval": "none claimed; point comparison",
        },
        "reproduction_gate": gate,
        "holdout_months": n_months,
        "holdout_month_labels": [label(m) for m in hold_months],
        "n_corridors": len(top),
        "macro": {
            "mase_seasonal_naive": r_naive,
            "mase_model": r_model,
            "mase_combination": r_comb,
            "combination_minus_naive": round(r_comb - r_naive, 6),
            "naive_minus_combination": round(margin, 6),
            "margin_percent_of_naive": round(pct, 4),
        },
        "corridors_beating_naive": wins,
        "corridors_total": len(top),
        "corridors": corridors,
        "decision": {
            "rule_restated": {
                "combination_beats_naive": "combination macro MASE < naive macro MASE",
                "combination_does_not_beat_naive": "combination macro MASE >= naive "
                                                   "macro MASE",
            },
            "call": branch,
        },
        "required_sentence": sentence,
        "caveats": CAVEATS,
        "check": {"command": "./.venv/bin/python scripts/corridor_combination.py --check",
                  "tolerance": TOL},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--check", action="store_true",
                    help="recompute deterministically and compare every numeric "
                         "leaf of the committed JSON at 1e-6")
    args = ap.parse_args()
    out_path = Path(args.out)

    saved = None
    if args.check:
        if not out_path.is_file():
            print(f"--check FAILED: {out_path} does not exist")
            return 1
        saved = json.loads(out_path.read_text())

    result = build()

    if args.check:
        a = dict(ce.numeric_leaves(saved))
        b = dict(ce.numeric_leaves(result))
        bad = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                bad.append((k, a.get(k), b.get(k)))
                continue
            if not math.isclose(a[k], b[k], rel_tol=TOL, abs_tol=TOL):
                bad.append((k, a[k], b[k]))
        if bad:
            print(f"--check FAILED: {len(bad)} numeric leaves differ (>{TOL}):")
            for k, va, vb in bad[:20]:
                print(f"  {k}: stored={va} recomputed={vb}")
            return 1
        if saved.get("required_sentence") != result["required_sentence"]:
            print("--check FAILED: required_sentence differs")
            return 1
        if saved.get("decision", {}).get("call") != result["decision"]["call"]:
            print("--check FAILED: decision call differs")
            return 1
        print(f"--check OK: {len(b)} numeric leaves match stored JSON at {TOL}, "
              "required_sentence and decision call identical")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} B)")
    g = result["reproduction_gate"]
    print(f"reproduction gate: ok={g['ok']} max_abs_diff={g['max_abs_diff']:.3e} "
          f"over {g['n_values_compared']} values")
    m = result["macro"]
    print(f"macro MASE  naive={m['mase_seasonal_naive']}  "
          f"model={m['mase_model']}  combination={m['mase_combination']}")
    print(f"corridors beating naive: {result['corridors_beating_naive']} of "
          f"{result['corridors_total']}")
    print(f"decision call: {result['decision']['call']}")
    print("required_sentence:", result["required_sentence"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
