#!/usr/bin/env python3
"""The control the whitespace signing head did not have.

Pre-registered in WHITESPACE-CONTROL-PREREG.md and committed BEFORE this producer
was written. Every arm, every metric, every threshold and every void condition in
this file is the one registered there. Read that file first.

WHAT THIS IS. results/whitespace.json ranks Singapore merchant-signing buckets by a
composite of four real observable signals and never states how the four are combined.
It is the only exhibit on the page with no control. This producer states the formula,
scores the SAME bucket set under seven simpler rankings, and measures how much of the
released list survives each one. It also carries three sanity rungs, because a pipeline
that cannot report both no difference and total difference reports nothing.

WHAT THIS IS NOT. It is not a validation. There is no observed merchant-acceptance
label in this corpus, so no arm here can be shown to predict a real signing, and no
proxy outcome is constructed. The exhibit gains a control and a stated formula. It does
not gain a ground truth. The acceptance-gap signal stays simulated and stays labelled.

NOT A SAFETY-BAND EXHIBIT. This file lives beside scripts/safety/privacy_ladder.py
because that producer prices the same frozen artifact and imports the same module. It is
NOT in inline_results.SAFETY_FILES and NOT in inline_results.EXHIBITS, so writing
results/whitespace_control.json cannot change the built page by itself.

NOTHING IS MOVED. scripts/whitespace_exhibit.py is imported and never edited, and no
leaf of results/whitespace.json is written by this producer. The rebuilt bucket set is
asserted identical to the committed released list before any control number is computed.

Laptop, CPU, deterministic, seed 42. --check recomputes every numeric leaf and compares
at 1e-6, in the shape scripts/safety/privacy_ladder.py --check uses.
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
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "fm"))

import whitespace_exhibit as ws          # noqa: E402  the frozen producer under control
from common import atomic_write_json, versions_dict  # noqa: E402

SEED = 42
RELEASED_ROWS = ws.MAX_RANKING_ROWS      # 400, the rows the page releases
TOP_KS = [20, 100, RELEASED_ROWS]
RANDOM_SEEDS = 20                        # same count as privacy_ladder's noise seeds

# Pre-registered decision thresholds (WHITESPACE-CONTROL-PREREG.md section 4).
DENSITY_ARMS = ["raw_venue_count", "density_channel"]
NULL_SPEARMAN_MIN = 0.95
NULL_TOP400_MIN = 380
NULL_TOP20_CHURN_MAX = 4
MATERIAL_SPEARMAN_MAX = 0.80
MATERIAL_TOP400_MAX = 320

# Pre-registered sanity gates (section 5).
RANDOM_MEAN_ABS_RHO_MAX = 0.10
RANDOM_SEED_ABS_RHO_MAX = 0.15
RANDOM_MEAN_TOP20_OVERLAP_MAX = 2.0
WIRE_CROSS_CHECK_TOL = 1e-4              # the published wire rhos are stored at 4 dp
REPRO_TOL = 1e-6

WEIGHTS_ARE = (
    "The four weights are a hand-set judgement call and not a fitted quantity. WEIGHTS at "
    "scripts/whitespace_exhibit.py:70 is a literal constant. Nothing in this repository fits it, "
    "tunes it, cross-validates it or selects it against an outcome, because there is no outcome to "
    "select it against. The four numbers were chosen by hand to sum to one."
)

NO_GROUND_TRUTH = (
    "There is no observed merchant-acceptance label in this corpus. Foursquare venue records say what "
    "a venue is and where it is, never whether it was approached, signed, or would sign. So no arm "
    "here is shown to predict a real signing, no proxy outcome is constructed, and rank agreement "
    "between two rankings is not evidence that either one is correct. This exhibit now has a control "
    "and a stated formula. It still has no outcome label, and the acceptance-gap signal in "
    "results/whitespace.json#simulation_params is simulated."
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ arms ----

def bucket_key(b: dict) -> str:
    return f"{b['cell_x']}_{b['cell_y']}_{b['category_group']}"


def arm_scores(buckets: list[dict], wire_by_group: dict[str, float]) -> dict[str, np.ndarray]:
    """Every arm scores the SAME bucket set, in the producer's own stable order.

    The composite is read back from the rebuilt buckets, so the reference arm is the
    shipped score itself and not a restatement of it.
    """
    f64 = lambda xs: np.asarray(xs, dtype=np.float64)  # noqa: E731
    mdr = f64([b["mdr_prior_mean"] for b in buckets])
    tz = f64([b["tz_mean"] for b in buckets])
    dens = f64([b["dens_mean"] for b in buckets])
    indep = f64([b["indep_share"] for b in buckets])
    return {
        "composite": f64([b["score_real_signals"] for b in buckets]),
        "raw_venue_count": f64([b["n_pois"] for b in buckets]),
        "density_channel": dens,
        "tourist_zone_channel": tz,
        "mdr_prior_channel": mdr,
        "independent_channel": indep,
        "equal_weights": 0.25 * (mdr + tz + dens + indep),
        "embedding_wire": f64([wire_by_group[b["category_group"]] for b in buckets]),
    }


def order_of(score: np.ndarray) -> np.ndarray:
    """Descending score, ties broken by the producer's own stable bucket order.

    Identical rule to scripts/whitespace_exhibit.py:719, which is how the page itself
    resolves a tie, so a top-k cut here is the cut the page would make.
    """
    return np.argsort(-score, kind="stable")


# --------------------------------------------------------------- metrics ----

def rho(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman on the SCORES, so scipy assigns average ranks to ties.

    Returns None when either side is constant on the compared set, which makes the
    correlation undefined rather than zero. A None prints as no value and no claim may
    be made from it.
    """
    if len(a) < 3 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return None
    v = float(spearmanr(a, b).statistic)
    return None if math.isnan(v) else v


def utility(ref: np.ndarray, arm: np.ndarray, keys: list[str]) -> dict:
    """Agreement between the composite ranking and one arm's ranking.

    spearman_full is over every bucket formed. spearman_released is restricted to the
    buckets the page actually releases, which is the composite's own top 400. The top-k
    numbers are set overlaps on the released list, which is what a partner acts on.
    """
    ref_order, arm_order = order_of(ref), order_of(arm)
    rel = ref_order[:RELEASED_ROWS]
    out = {
        "spearman_full": rho(ref, arm),
        "n_buckets_compared_full": int(len(ref)),
        "spearman_released": rho(ref[rel], arm[rel]),
        "n_buckets_compared_released": int(len(rel)),
    }
    for k in TOP_KS:
        r = {keys[i] for i in ref_order[:k]}
        a = {keys[i] for i in arm_order[:k]}
        out[f"top{k}_overlap"] = int(len(r & a))
    r20 = {keys[i] for i in ref_order[:20]}
    a20 = {keys[i] for i in arm_order[:20]}
    out["top20_left"] = int(len(r20 - a20))
    out["top20_entered"] = int(len(a20 - r20))
    out["top20_churn"] = out["top20_left"] + out["top20_entered"]
    return out


UTIL_KEYS = ["spearman_full", "spearman_released", "top20_overlap", "top100_overlap",
             f"top{RELEASED_ROWS}_overlap", "top20_churn"]


def spread(vals: list) -> dict:
    """Mean and percentile 95% spread across seeds, same shape privacy_ladder uses.

    A degenerate spread is null and no claim may be made from that row's width.
    """
    a = np.array([v for v in vals if v is not None], dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "spread": None, "n_seeds": 0}
    if a.size < 2:
        return {"mean": float(a[0]), "spread": None, "n_seeds": int(a.size)}
    return {"mean": float(a.mean()),
            "spread": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            "n_seeds": int(a.size)}


# ------------------------------------------------------------- decisions ----

def verdict_for(u: dict) -> str:
    """The pre-registered three-way call, applied to one arm and nothing else."""
    sf, t400, churn = u["spearman_full"], u[f"top{RELEASED_ROWS}_overlap"], u["top20_churn"]
    if sf is None:
        return "undefined: the arm is constant on the compared set"
    if sf >= NULL_SPEARMAN_MIN and t400 >= NULL_TOP400_MIN and churn <= NULL_TOP20_CHURN_MAX:
        return "reproduces the composite at the pre-registered null thresholds"
    if sf <= MATERIAL_SPEARMAN_MAX or t400 <= MATERIAL_TOP400_MAX:
        return "materially different from the composite at the pre-registered thresholds"
    return "between the two pre-registered thresholds; reported as obtained with no verdict word"


# ------------------------------------------------------------------ build ---

def build(data_path: Path, shipped_path: Path) -> dict:
    shipped = json.loads(shipped_path.read_text())
    wire_by_group = {r["group"]: float(r["cos_to_anchor"])
                     for r in shipped["wire"]["group_similarity"]}

    df, n_universe = ws.build_poi_frame(data_path)
    df = ws.compute_signals(df)
    buckets = ws.make_buckets(df)
    keys = [bucket_key(b) for b in buckets]
    scores = arm_scores(buckets, wire_by_group)
    ref = scores["composite"]
    ref_order = order_of(ref)

    # --- gate: the rebuilt set IS the shipped artifact, before any control number ---
    rebuilt_released = [buckets[int(i)] for i in ref_order[:RELEASED_ROWS]]
    committed = shipped["ranking"]
    label_mismatches = [
        {"rank": i + 1, "committed": committed[i]["bucket_label"], "rebuilt": rebuilt_released[i]["bucket_label"]}
        for i in range(min(len(committed), len(rebuilt_released)))
        if committed[i]["bucket_label"] != rebuilt_released[i]["bucket_label"]
    ]
    score_diffs = [abs(float(rebuilt_released[i]["score_real_signals"]) - float(committed[i]["score_real_signals"]))
                   for i in range(min(len(committed), len(rebuilt_released)))]
    repro = {
        "n_committed_released_rows": int(len(committed)),
        "n_rebuilt_released_rows": int(len(rebuilt_released)),
        "n_bucket_label_mismatches": int(len(label_mismatches)),
        "max_abs_score_diff": float(max(score_diffs)) if score_diffs else None,
        "tolerance": REPRO_TOL,
        "n_buckets_rebuilt": int(len(buckets)),
        "n_buckets_committed": int(shipped["n_buckets"]),
        "ok": bool(len(committed) == len(rebuilt_released) == RELEASED_ROWS
                   and not label_mismatches
                   and score_diffs and max(score_diffs) <= REPRO_TOL
                   and len(buckets) == int(shipped["n_buckets"])),
        "note": ("the control is measured against the artifact the page ships, and this is the "
                 "assertion that says so rather than the claim that says so"),
    }

    # --- the formula, with one bucket recomputed from its own published four numbers ---
    top = committed[0]
    s = top["signals"]
    hand = (ws.WEIGHTS["mdr_sensitivity_prior"] * s["mdr_sensitivity_prior"]
            + ws.WEIGHTS["tourist_zone_proximity"] * s["tourist_zone_score"]
            + ws.WEIGHTS["local_density"] * s["density_norm"]
            + ws.WEIGHTS["independent_share"] * s["independent_share"])
    formula = {
        "stated": ("score_real_signals(bucket) = 0.30*mean(mdr_sensitivity_prior) "
                   "+ 0.30*mean(tourist_zone_score) + 0.25*mean(density_norm) "
                   "+ 0.15*mean(independent_share)"),
        "weights": dict(ws.WEIGHTS),
        "weights_sum": float(sum(ws.WEIGHTS.values())),
        "weights_are": WEIGHTS_ARE,
        "why_the_bucket_form_is_exact": (
            "the per-venue score at scripts/whitespace_exhibit.py:361-364 is linear in the four "
            "channels and the bucket score at :396 is an unweighted mean over the venues in the "
            "bucket, so the weighted sum of the channel means equals the mean of the weighted sums"),
        "channels": {
            "mdr_sensitivity_prior": "per category group constant from MDR_PRIORS, six distinct values",
            "tourist_zone_score": "max over ten documented zones of exp(-d_km / 1.5) to the zone centroid",
            "density_norm": "min(log1p(venues within 250 m) / log1p(p99), 1.0), self excluded",
            "independent_share": "1 when the normalized venue name occurs fewer than 3 times in the slice, else 0",
        },
        "simulated_signal_is_not_in_the_formula": (
            "the demand-weighted acceptance-gap signal is simulated and enters only the separate "
            "sensitivity analysis at sensitivity(); it never enters score_real_signals or the "
            "published ranking order"),
        "worked_example": {
            "bucket_label": top["bucket_label"],
            "rank": int(top["rank"]),
            "published_signals": dict(s),
            "recomputed_by_hand_from_published_signals": round(float(hand), 6),
            "published_score_real_signals": float(top["score_real_signals"]),
            "abs_difference": float(abs(hand - float(top["score_real_signals"]))),
            "why_not_exactly_zero": ("the four signals are published rounded to four decimals, so a "
                                     "hand recomputation from the shipped file lands within about 1e-5"),
        },
    }

    # --- control arms ---------------------------------------------------------
    arm_defs = {
        "raw_venue_count": "bucket venue count, descending. No normalization, no KD-tree, no weights.",
        "density_channel": "the composite's own local-density channel alone.",
        "tourist_zone_channel": "the composite's own tourist-zone proximity channel alone.",
        "mdr_prior_channel": "the composite's own category prior alone. Six distinct values, mostly ties.",
        "independent_channel": "the composite's own independent-share channel alone.",
        "equal_weights": "the same four channels at 0.25 each. This arm prices the hand-set weights.",
        "embedding_wire": ("the category-level backbone column already published at "
                           "results/whitespace.json#ranking[].score_with_embeddings. Six distinct "
                           "values, read from the committed file rather than recomputed."),
    }
    arms = {}
    for name, desc in arm_defs.items():
        u = utility(ref, scores[name], keys)
        arms[name] = {
            "what": desc,
            "is_single_signal": name in ("density_channel", "tourist_zone_channel",
                                         "mdr_prior_channel", "independent_channel",
                                         "raw_venue_count"),
            "n_distinct_scores": int(len(np.unique(scores[name]))),
            "agreement_with_composite": u,
            "verdict": verdict_for(u),
        }

    # --- sanity rungs ---------------------------------------------------------
    u_identity = utility(ref, ref.copy(), keys)
    u_reversed = utility(ref, -ref, keys)
    rng_runs = []
    for i in range(RANDOM_SEEDS):
        r = np.random.default_rng(SEED + i).random(len(ref))
        rng_runs.append(utility(ref, r, keys))
    u_random = {k: spread([run[k] for run in rng_runs]) for k in UTIL_KEYS}
    random_abs_rhos = [abs(run["spearman_full"]) for run in rng_runs
                       if run["spearman_full"] is not None]

    identity_ok = bool(
        u_identity["spearman_full"] is not None and abs(u_identity["spearman_full"] - 1.0) <= REPRO_TOL
        and u_identity["spearman_released"] is not None
        and abs(u_identity["spearman_released"] - 1.0) <= REPRO_TOL
        and u_identity["top20_overlap"] == 20 and u_identity["top100_overlap"] == 100
        and u_identity[f"top{RELEASED_ROWS}_overlap"] == RELEASED_ROWS
        and u_identity["top20_churn"] == 0)
    reversed_ok = bool(
        u_reversed["spearman_full"] is not None and abs(u_reversed["spearman_full"] + 1.0) <= REPRO_TOL
        and u_reversed["top20_overlap"] == 0
        and u_reversed[f"top{RELEASED_ROWS}_overlap"] == 0)
    random_ok = bool(
        u_random["spearman_full"]["mean"] is not None
        and abs(u_random["spearman_full"]["mean"]) < RANDOM_MEAN_ABS_RHO_MAX
        and random_abs_rhos and max(random_abs_rhos) < RANDOM_SEED_ABS_RHO_MAX
        and u_random["top20_overlap"]["mean"] < RANDOM_MEAN_TOP20_OVERLAP_MAX)

    sanity = {
        "why": ("a pipeline that cannot report both no difference and total difference reports "
                "nothing, so neither a null nor a positive above means anything without these"),
        "identity": {
            "what": "the composite scored against itself; must agree perfectly",
            "agreement_with_composite": u_identity,
            "passes": identity_ok,
        },
        "reversed": {
            "what": "the composite negated; must disagree totally",
            "agreement_with_composite": u_reversed,
            "passes": reversed_ok,
        },
        "random_uniform": {
            "what": f"a seeded uniform random score per bucket, {RANDOM_SEEDS} seeds; must be close "
                    f"to independent of the composite",
            "seeds": [SEED + i for i in range(RANDOM_SEEDS)],
            "agreement_with_composite": u_random,
            "max_abs_spearman_full_across_seeds": float(max(random_abs_rhos)) if random_abs_rhos else None,
            "passes": random_ok,
            "spread_is": "across seeds, so it is mechanism randomness and not sampling uncertainty",
        },
        "deviation_from_the_pre_registration": (
            "section 5 asks for identity at spearman exactly 1.0 and reversed at exactly -1.0. Both "
            "land at magnitude 0.9999999999999999, which is the nearest double to 1 that scipy's "
            "rank correlation returns on a set this size. The gate is therefore applied at the same "
            "1e-6 tolerance every other numeric comparison in this project uses, and the raw values "
            "are stored above rather than rounded to hide the difference. Recorded here rather than "
            "smoothed over, because a pre-registration that gets quietly reinterpreted is worth "
            "nothing."),
        "thresholds": {
            "identity_tolerance": REPRO_TOL,
            "reversed_tolerance": REPRO_TOL,
            "random_mean_abs_spearman_full_max": RANDOM_MEAN_ABS_RHO_MAX,
            "random_per_seed_abs_spearman_full_max": RANDOM_SEED_ABS_RHO_MAX,
            "random_mean_top20_overlap_max": RANDOM_MEAN_TOP20_OVERLAP_MAX,
        },
    }

    # --- cross-check against a number already published ----------------------
    pub = shipped["wire"]["rank_reorder"]
    wire_u = arms["embedding_wire"]["agreement_with_composite"]
    d_rel = abs(wire_u["spearman_released"] - float(pub["spearman_ranked"])) \
        if wire_u["spearman_released"] is not None else None
    d_full = abs(wire_u["spearman_full"] - float(pub["spearman_all_buckets"])) \
        if wire_u["spearman_full"] is not None else None
    cross_check = {
        "what": ("the embedding_wire arm recomputes two correlations this project already published, "
                 "so the agreement code above is checked against a number it did not produce"),
        "published_spearman_ranked": float(pub["spearman_ranked"]),
        "recomputed_spearman_released": wire_u["spearman_released"],
        "abs_difference_released": d_rel,
        "published_spearman_all_buckets": float(pub["spearman_all_buckets"]),
        "recomputed_spearman_full": wire_u["spearman_full"],
        "abs_difference_full": d_full,
        "tolerance": WIRE_CROSS_CHECK_TOL,
        "why_this_tolerance": "the published values are stored rounded to four decimals",
        "ok": bool(d_rel is not None and d_full is not None
                   and d_rel <= WIRE_CROSS_CHECK_TOL and d_full <= WIRE_CROSS_CHECK_TOL),
    }

    # --- the pre-registered call ---------------------------------------------
    density = {a: arms[a]["agreement_with_composite"] for a in DENSITY_ARMS}
    strongest = max(DENSITY_ARMS,
                    key=lambda a: (density[a]["spearman_full"]
                                   if density[a]["spearman_full"] is not None else -2.0))
    null_hits = [a for a in DENSITY_ARMS
                 if density[a]["spearman_full"] is not None
                 and density[a]["spearman_full"] >= NULL_SPEARMAN_MIN
                 and density[a][f"top{RELEASED_ROWS}_overlap"] >= NULL_TOP400_MIN
                 and density[a]["top20_churn"] <= NULL_TOP20_CHURN_MAX]
    s_str = density[strongest]["spearman_full"]
    material = bool(s_str is not None
                    and (s_str <= MATERIAL_SPEARMAN_MAX
                         or density[strongest][f"top{RELEASED_ROWS}_overlap"] <= MATERIAL_TOP400_MAX))
    if null_hits:
        call = "null"
        headline = ("The composite is an expensive way to sort by density. The density arm "
                    + ", ".join(null_hits)
                    + " reproduces the shipped ranking at every pre-registered null threshold.")
    elif material:
        call = "material_difference"
        headline = (f"The composite is not a density sort. Its strongest density control, "
                    f"{strongest}, agrees at Spearman {s_str:.4f} over all "
                    f"{density[strongest]['n_buckets_compared_full']} buckets and keeps "
                    f"{density[strongest][f'top{RELEASED_ROWS}_overlap']} of the {RELEASED_ROWS} "
                    f"released rows. That says the four signals do something one column does not. It "
                    f"does not say the composite is right, because nothing here can say that.")
    else:
        call = "between_thresholds"
        headline = (f"Between the two pre-registered thresholds. The strongest density control, "
                    f"{strongest}, agrees at Spearman "
                    f"{s_str:.4f} over all buckets. Reported as obtained, with no verdict word.")

    # A restatement of arms already registered in section 2 of the pre-registration.
    # It adds no arm, no metric and no threshold. It exists so the two results that are
    # least flattering to the composite are in the summary and not only in the table.
    ew = arms["equal_weights"]["agreement_with_composite"]
    single = {a: arms[a]["agreement_with_composite"] for a in arm_defs
              if arms[a]["is_single_signal"]}
    closest_released = max(single, key=lambda a: (single[a]["spearman_released"]
                                                  if single[a]["spearman_released"] is not None else -2.0))
    cr = single[closest_released]
    other_arms = [
        (f"The hand-set weights are close to not load bearing. Scoring the same four channels at "
         f"0.25 each agrees with the shipped composite at Spearman {ew['spearman_full']:.4f} over "
         f"all {ew['n_buckets_compared_full']} buckets and keeps "
         f"{ew[f'top{RELEASED_ROWS}_overlap']} of the {RELEASED_ROWS} released rows, with a top-20 "
         f"churn of {ew['top20_churn']}. So the particular numbers 0.30, 0.30, 0.25 and 0.15 are "
         f"doing very little work. That is an argument that the weighting is not load bearing, and "
         f"it is not an argument that the weights are right."),
        (f"On the released list the closest single signal is {closest_released}, not density. It "
         f"agrees with the composite at Spearman {cr['spearman_released']:.4f} across the "
         f"{RELEASED_ROWS} released rows and keeps {cr[f'top{RELEASED_ROWS}_overlap']} of them, so "
         f"much of what a partner would receive is ordered by that one channel."),
        (f"The independent-share channel is close to independent of the composite at Spearman "
         f"{single['independent_channel']['spearman_full']:.4f} over all buckets, which is what a "
         f"0.15 weight on a binary channel buys."),
        ("None of this says any ranking here is accurate. There is still no outcome label."),
    ]

    decision = {
        "pre_registration": "WHITESPACE-CONTROL-PREREG.md, committed before this producer was written",
        "pre_registered_thresholds": {
            "density_arms": DENSITY_ARMS,
            "null_spearman_full_min": NULL_SPEARMAN_MIN,
            "null_top400_overlap_min": NULL_TOP400_MIN,
            "null_top20_churn_max": NULL_TOP20_CHURN_MAX,
            "material_spearman_full_max": MATERIAL_SPEARMAN_MAX,
            "material_top400_overlap_max": MATERIAL_TOP400_MAX,
        },
        "strongest_density_arm": strongest,
        "null_threshold_hits": null_hits,
        "call": call,
        "headline": headline,
        "what_the_other_arms_showed": other_arms,
        "what_the_other_arms_showed_is": (
            "a restatement of arms already registered in section 2 of the pre-registration. It adds "
            "no arm, no metric and no threshold, and it exists so the results least flattering to "
            "the composite sit in the summary and not only in the table."),
        "what_the_call_is_not": ("a statement that any ranking here is accurate. The comparison is "
                                 "agreement between rankings, and agreement is not accuracy."),
    }

    return {
        "seed": SEED,
        "versions": versions_dict(),
        "generated_by": "scripts/safety/whitespace_control.py --check-able",
        "data_sources": [
            ws.DATA_SOURCE,
            {"name": "One Loop whitespace ranking under control (frozen, not modified)",
             "url": "results/whitespace.json produced by scripts/whitespace_exhibit.py",
             "sha256": sha256_file(shipped_path)},
        ],
        "labels": ["real-signals base", "control arms", "no outcome label", "pseudonymized"],
        "what_this_is": (
            "the control the whitespace signing head did not have: the composite ranking measured "
            "against seven simpler rankings of the same bucket set, plus three sanity rungs"),
        "no_ground_truth": NO_GROUND_TRUTH,
        "not_a_page_exhibit": (
            "this file is in neither inline_results.EXHIBITS nor inline_results.SAFETY_FILES, so "
            "writing it cannot change the built page by itself"),
        "universe": shipped["universe"],
        "n_pois": int(n_universe),
        "n_pois_card_accepting_universe": int(df.height),
        "n_buckets": int(len(buckets)),
        "n_released_rows": RELEASED_ROWS,
        "reproduces_shipped": repro,
        "formula": formula,
        "metrics_note": (
            "spearman is computed on the SCORES, so ties take average ranks, which matters for the "
            "two arms carrying six distinct values across more than a thousand buckets. Top-k "
            "overlaps are set intersections on orderings whose ties are broken by the shipped stable "
            "order, so a cut here is the cut the page would make."),
        "arms": arms,
        "sanity": sanity,
        "cross_check_against_published": cross_check,
        "decision": decision,
        "check": {
            "command": "python3 scripts/safety/whitespace_control.py --check",
            "tolerance": REPRO_TOL,
        },
    }


# ------------------------------------------------------------------ check ---

SKIP_CHECK_PREFIXES = ()


def numeric_leaves(obj, path="") -> list[tuple[str, float]]:
    out = []
    if isinstance(obj, dict):
        for k in sorted(obj):
            if k == "versions":
                continue
            out.extend(numeric_leaves(obj[k], f"{path}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(numeric_leaves(v, f"{path}/{i}"))
    elif isinstance(obj, bool):
        out.append((path, float(obj)))
    elif isinstance(obj, (int, float)):
        out.append((path, float(obj)))
    return out


def compare(fresh: dict, stored: dict, tol: float) -> int:
    a = dict(numeric_leaves(fresh))
    b = dict(numeric_leaves(stored))
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


def gates(out: dict) -> list[str]:
    """Every void condition from the pre-registration, section 5."""
    fails = []
    if not out["reproduces_shipped"]["ok"]:
        fails.append("the rebuilt bucket set does not reproduce the committed released list")
    for rung in ("identity", "reversed", "random_uniform"):
        if not out["sanity"][rung]["passes"]:
            fails.append(f"sanity rung {rung} failed")
    if not out["cross_check_against_published"]["ok"]:
        fails.append("the embedding_wire arm does not reproduce the published wire correlations")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data" / "fsq_sg.parquet"))
    ap.add_argument("--shipped", default=str(REPO / "results" / "whitespace.json"))
    ap.add_argument("--out", default=str(REPO / "results" / "whitespace_control.json"))
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare every numeric leaf at 1e-6; exit 0/5")
    ap.add_argument("--check-tol", type=float, default=REPRO_TOL)
    args = ap.parse_args()

    data_path = Path(args.data)
    got = sha256_file(data_path)
    if got != ws.DATA_SOURCE["sha256"]:
        sys.exit(f"ERROR: sha256 mismatch for {data_path}: {got}")

    out = build(data_path, Path(args.shipped))

    fails = gates(out)
    if fails:
        for f in fails:
            print(f"GATE FAILED: {f}")
        print("VOID: the pre-registration says the file is not written when a gate fails, and the "
              "finding is that this producer cannot be trusted rather than anything about the composite")
        return 4

    if args.check:
        stored = json.loads(Path(args.out).read_text())
        return compare(out, stored, args.check_tol)

    atomic_write_json(args.out, out)
    print(f"\nwrote {args.out}")
    print(f"  reproduces shipped released list: {out['reproduces_shipped']['ok']} "
          f"(max abs score diff {out['reproduces_shipped']['max_abs_score_diff']:.2e})")
    print(f"  formula: {out['formula']['stated']}")
    we = out["formula"]["worked_example"]
    print(f"  worked example rank {we['rank']}: by hand {we['recomputed_by_hand_from_published_signals']} "
          f"vs published {we['published_score_real_signals']} (diff {we['abs_difference']:.2e})")
    print(f"  {out['n_buckets']:,} buckets, {out['n_released_rows']} released")
    for name, a in out["arms"].items():
        u = a["agreement_with_composite"]
        sf = "none" if u["spearman_full"] is None else f"{u['spearman_full']:+.4f}"
        sr = "none" if u["spearman_released"] is None else f"{u['spearman_released']:+.4f}"
        print(f"  {name:<22} rho_full {sf}  rho_released {sr}  "
              f"top20 {u['top20_overlap']:>3}/20  top100 {u['top100_overlap']:>3}/100  "
              f"top400 {u[f'top{RELEASED_ROWS}_overlap']:>3}/400  churn {u['top20_churn']:>2}")
    for rung in ("identity", "reversed"):
        u = out["sanity"][rung]["agreement_with_composite"]
        print(f"  SANITY {rung:<14} rho_full {u['spearman_full']:+.4f}  "
              f"top20 {u['top20_overlap']}/20  top400 {u[f'top{RELEASED_ROWS}_overlap']}/400  "
              f"passes {out['sanity'][rung]['passes']}")
    ru = out["sanity"]["random_uniform"]["agreement_with_composite"]
    print(f"  SANITY random_uniform  rho_full mean {ru['spearman_full']['mean']:+.4f} "
          f"spread {ru['spearman_full']['spread']}  top20 mean {ru['top20_overlap']['mean']:.2f}  "
          f"passes {out['sanity']['random_uniform']['passes']}")
    cc = out["cross_check_against_published"]
    print(f"  CROSS-CHECK wire: released {cc['recomputed_spearman_released']:+.4f} vs published "
          f"{cc['published_spearman_ranked']:+.4f}, full {cc['recomputed_spearman_full']:+.4f} vs "
          f"published {cc['published_spearman_all_buckets']:+.4f}, ok {cc['ok']}")
    print(f"  CALL: {out['decision']['call']}")
    print(f"  {out['decision']['headline']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
