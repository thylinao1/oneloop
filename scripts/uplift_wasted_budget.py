#!/usr/bin/env python3
"""
uplift_wasted_budget.py: measure the wasted-offer-budget claim on our own randomized data.

Why this exists
---------------
Wasted offer budget is the reason this product exists, and the document argued it from other
people's papers while sitting on a randomized experiment that can answer it directly. The
Hillstrom segment table in results/uplift.json already carries, per segment, the treated count,
the rank a response model gives it, and the rank the randomized arms give its measured uplift.
Crossing those two answers the question in one line: of the customers a response model would
have mailed, what share sit in segments the experiment measures as worst?

No new measurement. Every number is a count or a ratio over the committed segment table.

On the split, which is a choice and is therefore made visible
------------------------------------------------------------
"Top" and "bottom" need a cut, and the cut changes the answer. This file reports halves,
terciles and quartiles together rather than one, because a single number here would be a
number chosen after seeing all three. The tercile is what the page prints. It is not the
flattering one: the half split gives a much larger wasted share, and quoting that instead
would have been the easy move.

Run: python3 scripts/uplift_wasted_budget.py    writes results/uplift_wasted_budget.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derivation_check import write_or_check

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "uplift.json"
OUT = ROOT / "results" / "uplift_wasted_budget.json"

SPLITS = (("half", 2), ("tercile", 3), ("quartile", 4))


def main() -> int:
    d = json.loads(SRC.read_text())
    segs = d["hillstrom"]["segments"]
    n = len(segs)
    need = ("n_treated", "response_rank", "measured_uplift_rank", "name")
    for s in segs:
        missing = [k for k in need if k not in s]
        if missing:
            print(f"REFUSING: segment {s.get('name')!r} is missing {missing}")
            return 1

    best = min(segs, key=lambda s: s["measured_uplift_rank"])

    out = {}
    for label, parts in SPLITS:
        k = n // parts
        if k < 1:
            continue
        top = [s for s in segs if s["response_rank"] <= k]
        bottom = {s["name"] for s in segs if s["measured_uplift_rank"] > n - k}
        mailed = sum(s["n_treated"] for s in top)
        wasted = sum(s["n_treated"] for s in top if s["name"] in bottom)
        overlap = [
            {
                "segment": s["name"],
                "n_treated": s["n_treated"],
                "response_rank": s["response_rank"],
                "measured_uplift_rank": s["measured_uplift_rank"],
                "measured_visit_uplift_pp": s["measured_visit_uplift_pp"],
            }
            for s in top if s["name"] in bottom
        ]
        out[label] = {
            "group_size_segments": k,
            "treated_in_response_top_group": mailed,
            "treated_also_in_measured_uplift_bottom_group": wasted,
            "wasted_share": wasted / mailed if mailed else None,
            "wasted_percent": 100.0 * wasted / mailed if mailed else None,
            "segments_in_both": overlap,
        }

    payload = {
        "what_this_is": (
            "The wasted-offer-budget claim measured on the randomized Hillstrom arms instead of "
            "argued from citations. For each split, the customers a response model ranks in its "
            "top group are crossed against the segments the randomized arms measure in the bottom "
            "group of actual uplift. No new measurement: every value is a count or ratio over the "
            "committed segment table in results/uplift.json."
        ),
        "derived_from": {"file": "results/uplift.json", "pointer": "/hillstrom/segments"},
        "generated_by": "scripts/uplift_wasted_budget.py",
        "labels": d.get("labels", []),
        "n_segments": n,
        # Bound here rather than by array index so the comparison the page prints cannot silently
        # point at a different segment if the source table is ever reordered.
        "best_measured_uplift": {
            "segment": best["name"],
            "measured_visit_uplift_pp": best["measured_visit_uplift_pp"],
            "measured_uplift_rank": best["measured_uplift_rank"],
        },
        "the_split_is_a_choice": (
            "Top and bottom need a cut and the cut moves the answer, so all three cuts are "
            "reported here and the page prints the tercile. The tercile is not the flattering "
            "one: the half split gives a much larger wasted share and would have been the easy "
            "number to quote."
        ),
        "domain_caveat": (
            "Hillstrom is a retail e-commerce e-mail experiment. The outcome is a site visit, not "
            "billed business on a card, and the segments are that retailer's, not a card "
            "portfolio's. This measures the mechanism, not the size of the opportunity at Amex."
        ),
        "splits": out,
    }
    rc = write_or_check(OUT, payload)
    if rc or "--check" in sys.argv[1:]:
        return rc
    for label, v in out.items():
        print(f"  {label:9} {v['treated_also_in_measured_uplift_bottom_group']:5} of "
              f"{v['treated_in_response_top_group']:5} treated = {v['wasted_percent']:.2f} percent")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
