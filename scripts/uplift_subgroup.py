#!/usr/bin/env python3
"""
uplift_subgroup.py: the fairness-adjacent measurement the obligation map owes.

Why this exists
---------------
Section 4 maps the design to MAS FEAT, whose first principle is fairness, and the document
carried no fairness measurement of any kind and no statement of why. In a section built to
separate what we mapped from what we measured, that is the one gap the section exists to catch.

The honest position on protected attributes, which is why this measures a proxy and says so:
none of the four corpora carries a protected attribute that can be read.
  * IBM TabFormer is synthetic.
  * Criteo Uplift v2.1 ships anonymized features f0 to f11.
  * dunnhumby Complete Journey's demographic file was anonymized in its 2023 re-release into
    opaque classification_1 through classification_5 columns (see cj_common.py), so even the
    one asset whose name suggests demographics cannot be read as one.
  * Hillstrom carries a three-level zip-code class: Rural, Suburban, Urban.

That zip class is a GEOGRAPHIC PROXY and not a protected attribute. It is measured here because
it is the only group variable in the project that sits behind a randomized assignment, and
because a proxy measured honestly is worth more than a protected-attribute number that would
have to be manufactured.

What is measured, on the randomized arms
----------------------------------------
1. The effect itself, per group, as a direct arm contrast with NO model in it: treated visit
   rate minus control visit rate. A difference here is a fact about where the campaign works,
   not evidence of unfairness.
2. Selection rate per group at the top decile, under uplift ranking and under response ranking.
   This is the fairness question that matters for a targeting rule: who does the rule choose.
   A rule may legitimately select a group less often BECAUSE the measured effect there is
   smaller. It is when selection and measured effect disagree that there is something to answer
   for, so both are reported side by side and the disagreement is computed rather than asserted.

The models are in-sample, exactly as the shipped Hillstrom exhibit already flags, so the
selection rates are illustrative of the mechanism rather than an out-of-sample claim.

Run: python3 scripts/uplift_subgroup.py            writes results/uplift_subgroup.json
     python3 scripts/uplift_subgroup.py --check    recompute and compare at 1e-6
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derivation_check import write_or_check  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "uplift_subgroup.json"
CSV = ROOT / "data" / "hillstrom.csv"

TOP_FRACTION = 0.10  # the same top decile the shipped uplift exhibit reports
BOOT = 2000
SEED = 7


def main() -> int:
    if not CSV.is_file():
        print(f"REFUSING: {CSV.relative_to(ROOT)} is not present. It is gitignored like every "
              f"raw corpus in this project; scripts/uplift_exhibit.py downloads it with a "
              f"checksum. Nothing is written.")
        return 1

    import polars as pl
    from uplift_exhibit import hillstrom_features, x_learner_scores

    df = pl.read_csv(CSV, schema_overrides={"spend": pl.Float64, "history": pl.Float64},
                     infer_schema_length=10000)
    df = df.filter(pl.col("segment").is_in(["Womens E-Mail", "No E-Mail"]))
    treat = (df["segment"] == "Womens E-Mail").cast(pl.Int8).to_numpy()
    visit = df["visit"].to_numpy().astype(np.int8)
    zips = np.array(df["zip_code"].to_list())
    X, _ = hillstrom_features(df)

    it, ic = np.flatnonzero(treat == 1), np.flatnonzero(treat == 0)
    e = len(it) / (len(it) + len(ic))
    resp, cate = x_learner_scores(X[it], visit[it], X[ic], visit[ic], X, e, threads=1)

    # Top-decile selection under each ranking rule, over the whole population.
    k = int(len(cate) * TOP_FRACTION)
    sel_uplift = np.zeros(len(cate), dtype=bool)
    sel_uplift[np.argsort(-cate, kind="stable")[:k]] = True
    sel_resp = np.zeros(len(resp), dtype=bool)
    sel_resp[np.argsort(-resp, kind="stable")[:k]] = True

    rng = np.random.default_rng(SEED)
    groups = []
    for name in sorted(set(zips.tolist())):
        m = zips == name
        mt, mc = m & (treat == 1), m & (treat == 0)
        nt, nc = int(mt.sum()), int(mc.sum())
        vt, vc = float(visit[mt].mean()), float(visit[mc].mean())
        # Bootstrap the arm contrast inside the group, resampling each arm independently,
        # which is what the randomization licenses.
        t_idx, c_idx = np.flatnonzero(mt), np.flatnonzero(mc)
        draws = np.empty(BOOT)
        for b in range(BOOT):
            a = visit[rng.choice(t_idx, nt, replace=True)].mean()
            d = visit[rng.choice(c_idx, nc, replace=True)].mean()
            draws[b] = (a - d) * 100.0
        lo, hi = np.percentile(draws, [2.5, 97.5])
        groups.append({
            "group": name,
            "n_total": int(m.sum()),
            "share_of_population": float(m.sum()) / len(zips),
            "n_treated": nt,
            "n_control": nc,
            "measured_visit_uplift_pp": (vt - vc) * 100.0,
            "measured_visit_uplift_ci_pp": [float(lo), float(hi)],
            "interval_clears_zero": bool(lo > 0 or hi < 0),
            "selection_rate_uplift_ranking": float(sel_uplift[m].mean()),
            "selection_rate_response_ranking": float(sel_resp[m].mean()),
        })

    # Selection rate parity: the ratio of the smallest to the largest group selection rate.
    # 1.0 is exact parity. Reported for both rules so the comparison is like for like.
    for rule in ("uplift", "response"):
        key = f"selection_rate_{rule}_ranking"
        rates = [g[key] for g in groups]
        lo_r, hi_r = min(rates), max(rates)
        for g in groups:
            g[f"{rule}_selection_vs_highest_group"] = (g[key] / hi_r) if hi_r > 0 else None

    rates_u = [g["selection_rate_uplift_ranking"] for g in groups]
    rates_r = [g["selection_rate_response_ranking"] for g in groups]
    eff = [g["measured_visit_uplift_pp"] for g in groups]
    order_by_effect = [g["group"] for g in sorted(groups, key=lambda g: -g["measured_visit_uplift_pp"])]
    order_by_selection = [g["group"] for g in sorted(groups, key=lambda g: -g["selection_rate_uplift_ranking"])]

    payload = {
        "what_this_is": (
            "The fairness-adjacent measurement the Section 4 obligation map owes, measured on the "
            "randomized Hillstrom arms rather than asserted. Two things per group: the effect "
            "itself as a direct arm contrast with no model in it, and the rate at which each "
            "targeting rule selects that group into the top decile. A rule may legitimately "
            "select a group less often because the measured effect there is smaller; it is when "
            "selection and measured effect DISAGREE that there is something to answer for, so "
            "both are reported side by side."
        ),
        "generated_by": "scripts/uplift_subgroup.py",
        "labels": ["real", "randomized-experiment", "proxy: geography is not a protected attribute"],
        "the_group_variable_is_a_proxy": (
            "zip_code is a three-level geographic class (Rural, Suburban, Urban), NOT a protected "
            "attribute. It is measured because it is the only group variable in this project "
            "sitting behind a randomized assignment. None of the four corpora carries a readable "
            "protected attribute: TabFormer is synthetic, Criteo ships anonymized f0 to f11, and "
            "Complete Journey's demographic file was anonymized into opaque classifications in "
            "its 2023 re-release. A protected-attribute fairness number computed here would be "
            "manufactured, and this file exists so that none is."
        ),
        "models_are_in_sample": (
            "The uplift and response scores come from the same in-sample X-learner the shipped "
            "Hillstrom exhibit uses and flags as illustrative. The selection rates therefore "
            "describe the mechanism, not an out-of-sample guarantee. The measured effect column "
            "has no model in it at all and is not affected."
        ),
        "domain_caveat": (
            "Hillstrom is a retail e-commerce e-mail experiment. The outcome is a site visit, not "
            "billed business on a card, and the geography is that retailer's customer base."
        ),
        "top_fraction": TOP_FRACTION,
        "bootstrap_draws": BOOT,
        "seed": SEED,
        "n_groups": len(groups),
        # Named keys as well as the list, so the page binds to a group by NAME and cannot
        # silently point at a different one if the ordering ever changes. Binding a printed
        # figure to an array index is a defect this project has already had once.
        "by_group": {g["group"]: g for g in groups},
        "lowest_effect_group": min(groups, key=lambda g: g["measured_visit_uplift_pp"])["group"],
        "lowest_effect_pp": min(g["measured_visit_uplift_pp"] for g in groups),
        "most_selected_group_response_ranking": max(
            groups, key=lambda g: g["selection_rate_response_ranking"])["group"],
        "response_ranking_over_selects_the_lowest_effect_group": (
            min(groups, key=lambda g: g["measured_visit_uplift_pp"])["group"]
            == max(groups, key=lambda g: g["selection_rate_response_ranking"])["group"]),
        "groups": groups,
        "group_order_by_measured_effect": order_by_effect,
        "group_order_by_uplift_selection": order_by_selection,
        "selection_and_effect_agree_on_order": order_by_effect == order_by_selection,
        "min_selection_rate_uplift": min(rates_u),
        "max_selection_rate_uplift": max(rates_u),
        "selection_parity_ratio_uplift": min(rates_u) / max(rates_u) if max(rates_u) else None,
        "selection_parity_ratio_response": min(rates_r) / max(rates_r) if max(rates_r) else None,
        "spread_in_measured_effect_pp": max(eff) - min(eff),
        "what_phase_1_owes": (
            "Subgroup lift and selection rates on the partner's own protected attributes, "
            "pre-registered before the first campaign opens and held to the same paired-interval "
            "standard as every comparison in this document, reported whichever way they move."
        ),
    }
    rc = write_or_check(OUT, payload)
    if rc or "--check" in sys.argv[1:]:
        return rc
    for g in groups:
        print(f"  {g['group']:10} n={g['n_total']:6}  effect {g['measured_visit_uplift_pp']:+.2f}pp "
              f"CI [{g['measured_visit_uplift_ci_pp'][0]:+.2f}, {g['measured_visit_uplift_ci_pp'][1]:+.2f}]"
              f"  selected {g['selection_rate_uplift_ranking'] * 100:5.2f}% (uplift) "
              f"{g['selection_rate_response_ranking'] * 100:5.2f}% (response)")
    print(f"selection parity ratio, uplift ranking: {payload['selection_parity_ratio_uplift']:.4f}")
    print(f"order agrees with measured effect: {payload['selection_and_effect_agree_on_order']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
