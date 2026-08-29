#!/usr/bin/env python3
"""
uplift_shares.py: the headline restated in the unit a budget owner actually uses.

Why this exists
---------------
The strongest number in the entry is a paired difference of 0.011049 with an interval entirely
above zero. It is the most defensible figure on the page and the least legible one: nobody runs a
campaign in units of incremental site-visit rate per customer targeted.

The same committed numbers restate as budget efficiency. Targeting a tenth of customers by
predicted incremental effect captures 59.3 percent of every incremental visit the campaign
produced. Response ranking at the same reach captures 48.6 percent. That is the question a
marketing decision scientist is actually asking, and both numbers come from the same file.

This is a restatement and not a new claim. The identity is exact: the top-decile effect divided
by the average effect, which the page already prints as 5.9 times, divided by ten reach, IS the
share of the total. cate_over_ate / 10 == 0.5930677 to the last digit. So the same caveat travels
with it: a ratio of point estimates carrying no interval of its own, with the paired difference
and its interval as the interval-bearing claim beside it.

Added after an internal review (2026-08-24) that asked for the shares to be
derived, bound and check-able rather than left as the prose "close to six tenths" the hero
carried, which a judge cannot click and which is vaguer than everything around it.

Run: python3 scripts/uplift_shares.py            writes results/uplift_shares.json
     python3 scripts/uplift_shares.py --check    recompute and compare at 1e-6
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derivation_check import write_or_check  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "uplift.json"
OUT = ROOT / "results" / "uplift_shares.json"

K_PERCENT = 10  # the top decile, the same depth the shipped exhibit reports
Z95 = 1.96  # two-sided 95 percent normal quantile, for the segment rows that store an SE


def _decision_reversal(d: dict) -> dict:
    """The wasted-budget row and the hidden-gem row from the Hillstrom segment table."""
    segs = d["hillstrom"]["segments"]
    waste = min((s for s in segs if s.get("verdict") == "wasted-budget"),
                key=lambda s: s["measured_spend_uplift_usd"])
    gem = max((s for s in segs if s.get("verdict") == "hidden-gem"),
              key=lambda s: s["measured_visit_uplift_pp"])
    keep = ("name", "response_rank", "n_treated", "measured_visit_uplift_pp",
            "measured_visit_uplift_se_pp", "measured_spend_uplift_usd",
            "measured_spend_uplift_se_usd")

    def with_intervals(s: dict) -> dict:
        # The segment table stores a point and a standard error per row and no bootstrap, so
        # the interval here is the normal approximation, point plus or minus Z95 standard
        # errors, and the page says so where it prints it. An earlier version of the first
        # page printed the point plus or minus ONE standard error, which reads as an interval
        # and is not one, and printed a spend figure whose interval spans zero as if it were a
        # measured fact. Both are what these fields exist to prevent.
        row = {k: s[k] for k in keep}
        v, se = s["measured_visit_uplift_pp"], s["measured_visit_uplift_se_pp"]
        row["measured_visit_uplift_ci95_pp"] = [v - Z95 * se, v + Z95 * se]
        row["visit_interval_spans_zero"] = (v - Z95 * se) <= 0.0 <= (v + Z95 * se)
        sv, sse = s["measured_spend_uplift_usd"], s["measured_spend_uplift_se_usd"]
        row["measured_spend_uplift_ci95_usd"] = [sv - Z95 * sse, sv + Z95 * sse]
        row["spend_interval_spans_zero"] = (sv - Z95 * sse) <= 0.0 <= (sv + Z95 * sse)
        return row

    return {
        "what_this_is": (
            "The two segments from the Hillstrom table that state the product in two rows: the one "
            "a response model buys whose randomized arms cannot tell its uplift from zero on either "
            "endpoint, and the one it skips whose visit uplift clears zero. Same committed table, "
            "same seed, no new run; stored under named keys so the page binds them without an "
            "array index. Intervals are the normal approximation on the stored standard error."
        ),
        "interval_method": "normal approximation, point plus or minus 1.96 standard errors",
        "n_segments": len(segs),
        "response_model_buys_this": with_intervals(waste),
        "response_model_skips_this": with_intervals(gem),
    }


def main() -> int:
    d = json.loads(SRC.read_text())
    visit = d["criteo"]["targeting_at_k"]["outcomes"]["visit"]
    k = visit["k"][str(K_PERCENT)]
    ate = visit["ate"]["value"]
    reach = K_PERCENT / 100.0

    cate_rate = k["cate_ranking"]["value"]
    resp_rate = k["response_ranking"]["value"]
    if ate <= 0:
        print(f"REFUSING: the whole-holdout average effect is {ate}, so a share of it is not defined.")
        return 1

    cate_share = cate_rate * reach / ate
    resp_share = resp_rate * reach / ate

    # The identity that makes this a restatement rather than a new measurement. If it ever stops
    # holding, the arithmetic here has drifted from the figure the page already prints and this
    # file must not be published as a restatement of it.
    identity = k["cate_over_ate"] * reach
    if abs(identity - cate_share) > 1e-9:
        print(f"REFUSING: cate_over_ate x reach = {identity} does not equal the computed share "
              f"{cate_share}. These are supposed to be the same number by construction.")
        return 1

    payload = {
        "what_this_is": (
            "The top-decile uplift headline restated as a share of the campaign's total incremental "
            "effect, which is the unit a budget owner works in. A restatement, not a new claim: it "
            "is the 5.9 times figure the page already prints, taken at one tenth of the reach."
        ),
        "derived_from": {
            "file": "results/uplift.json",
            "pointers": {
                "top_decile_rate_uplift_ranking": "/criteo/targeting_at_k/outcomes/visit/k/10/cate_ranking/value",
                "top_decile_rate_response_ranking": "/criteo/targeting_at_k/outcomes/visit/k/10/response_ranking/value",
                "whole_holdout_average_effect": "/criteo/targeting_at_k/outcomes/visit/ate/value",
            },
        },
        "generated_by": "scripts/uplift_shares.py",
        "labels": d.get("labels", []),
        "reach_percent": K_PERCENT,
        "whole_holdout_average_effect": ate,
        "top_decile_rate_uplift_ranking": cate_rate,
        "top_decile_rate_response_ranking": resp_rate,
        "share_of_incremental_effect_uplift_ranking": cate_share,
        "share_of_incremental_effect_response_ranking": resp_share,
        "share_percent_uplift_ranking": cate_share * 100.0,
        "share_percent_response_ranking": resp_share * 100.0,
        "share_gain_percentage_points": (cate_share - resp_share) * 100.0,
        # The two Hillstrom rows that make the case, stored under NAMED keys so page one can
        # bind them without an array index (a printed figure bound to an array position is a
        # defect this project has already had once). Same pick rule as the Section 6 callout:
        # the segment a response model buys that measurement values at nothing, and the one it
        # skips that measurement values highest.
        "decision_reversal": _decision_reversal(d),
        "carries_the_same_caveat": (
            "A ratio of point estimates with no interval of its own. The interval-bearing claim "
            "beside it is the paired difference between the two rankings at this depth, "
            "0.011049 with a 95 percent interval of [0.008545, 0.014453], entirely above zero. "
            "The corpus is a public advertising experiment rather than card data, and visit was "
            "the pre-registered robustness endpoint, not the primary."
        ),
        "identity_check": (
            "cate_over_ate x reach equals the uplift share exactly, which is what makes this a "
            "restatement of a figure already on the page rather than a second measurement."
        ),
    }
    rc = write_or_check(OUT, payload)
    if rc or "--check" in sys.argv[1:]:
        return rc
    print(f"at {K_PERCENT} percent reach: uplift ranking captures {cate_share * 100:.2f} percent of "
          f"the total incremental effect, response ranking {resp_share * 100:.2f} percent "
          f"({(cate_share - resp_share) * 100:.2f} points more)")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
