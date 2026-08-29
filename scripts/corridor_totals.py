#!/usr/bin/env python3
"""
corridor_totals.py: total the corridor errors in arrivals, which is the view that reverses.

Why this exists
---------------
The corridor result the page reports is macro MASE: the seasonal-naive baseline at 0.5302, the
pre-registered equal-weight blend at 0.5061. MASE is scale-free and the macro figure averages
the twelve corridors with equal weight, so a large corridor and a small one count the same.

results/corridor_combination.json also carries a per-corridor mae column, in arrivals. Summed
across the twelve, the blend is WORSE than the naive it is added to. That column was in the file
the page cites and the page never printed it, in a document whose stated premise is that it
prints the results that go against it. This file makes the reversing view a first-class number
so it can be printed and bound like any other.

No new measurement and no assumption: every value is a sum or a ratio of numbers already stored
in results/corridor_combination.json under /corridors. The macro MASE win is not withdrawn by
this and is not meant to be. The two views answer different questions, one weighting corridors
equally and one weighting arrivals equally, and a reader is entitled to both before deciding
what the blend is worth.

Run: python3 scripts/corridor_totals.py     writes results/corridor_totals.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derivation_check import write_or_check

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "corridor_combination.json"
OUT = ROOT / "results" / "corridor_totals.json"


def main() -> int:
    d = json.loads(SRC.read_text())
    rows = d["corridors"]
    if not rows:
        print("REFUSING: no corridor rows in the source file")
        return 1

    need = ("mae_seasonal_naive", "mae_model", "mae_combination")
    missing = [r.get("origin") for r in rows if any(k not in r for k in need)]
    if missing:
        print(f"REFUSING: corridor rows missing an mae column: {missing}")
        return 1

    naive = sum(r["mae_seasonal_naive"] for r in rows)
    model = sum(r["mae_model"] for r in rows)
    blend = sum(r["mae_combination"] for r in rows)
    beats = sum(1 for r in rows if r.get("combination_beats_naive"))

    payload = {
        "what_this_is": (
            "The corridor comparison totalled in arrivals rather than averaged scale-free. Every "
            "value here is a sum or a ratio of the per-corridor mae columns already stored in "
            "results/corridor_combination.json under /corridors, so this file adds no measurement "
            "and no assumption. It exists because this is the view that reverses the macro-MASE "
            "result, and the column was in the cited file without being printed."
        ),
        "derived_from": {"file": "results/corridor_combination.json", "pointer": "/corridors"},
        "generated_by": "scripts/corridor_totals.py",
        "labels": d.get("labels", []),
        "holdout": "2025-01 to 2026-01, 13 months across 12 corridors, the same holdout as the source",
        "n_corridors": len(rows),
        "total_absolute_error_arrivals": {
            "seasonal_naive": naive,
            "model_alone": model,
            "combination": blend,
        },
        "combination_minus_naive": blend - naive,
        "combination_worse_by_fraction_of_naive": (blend - naive) / naive,
        "corridors_where_combination_beats_naive": beats,
        "how_to_read_it": (
            "Macro MASE weights each corridor equally and the blend wins on it. Total absolute "
            "error weights each arrival equally and the naive wins on it, by "
            f"{blend - naive:.2f} arrivals over the holdout. The blend beats the naive in {beats} "
            f"of {len(rows)} corridors. Neither view is the single right one, and the entry does "
            "not claim the corridor lane on the strength of the one that flatters it."
        ),
    }
    rc = write_or_check(OUT, payload)
    if rc == 0 and "--check" not in sys.argv[1:]:
        print(f"wrote {OUT.relative_to(ROOT)}: naive {naive:,.2f} vs combination {blend:,.2f} "
              f"arrivals of total absolute error, combination worse by {blend - naive:,.2f} "
              f"({(blend - naive) / naive * 100:.2f} percent), beating naive in {beats} of {len(rows)}.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
