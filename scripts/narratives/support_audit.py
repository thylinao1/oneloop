#!/usr/bin/env python3
"""
support_audit.py: how much weight the loose part of layer one is actually carrying.

Why this exists
---------------
Layer one asks one question: is this numeral carried by some fact in the bundle. It says
nothing about whether the numeral was bound to the RIGHT fact, and a reader is entitled to
ask "how often does that go wrong, and how would you know". The honest answer is that there
is no semantic error rate here: layer one matches numerals and does not read sentences, and
no labelled set for the noun-binding question has been built.

What can be measured is how much of the passing is done by the strict part of the match and
how much by the loose part. `_matches` accepts a numeral three ways: the fact exactly as
stored, the fact rounded to the precision the narrative printed, and the fact read as a
percentage (a factor of one hundred either way). The last two are where a wrong binding can
hide: rounding is a half-unit window, so a stored 1.5 carries a printed 2, and the percentage
reading moves the decimal point. This counts every numeral in every committed narrative by the
closest way it is supported, so the widening is priced rather than assumed.

CPU only, no model, no network. Reads the committed narratives file and nothing else.

Usage:
  python3 scripts/narratives/support_audit.py --emit results/narratives_support.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "narratives"))

from faithcheck import (  # noqa: E402
    _NUM_RE,
    _matches,
    collect_fact_numbers,
    support_rank,
)

NARRATIVES = ROOT / "results" / "narratives.json"

RANK_LABELS = {
    0: "the fact exactly as stored",
    1: "the fact rounded to the precision the narrative printed",
    2: "the fact read as a percentage",
    3: "read as a percentage and rounded as well",
}


def audit(examples: list[dict]) -> dict:
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    unsupported = 0
    total = 0
    examples_with_scaled: list[str] = []

    for example in examples:
        facts = example.get("input_facts") or {}
        pool = collect_fact_numbers(facts)
        narrative = example.get("narrative") or ""
        scaled_here = False
        for match in _NUM_RE.finditer(narrative):
            raw = match.group(0)
            value = float(raw.replace(",", ""))
            decimals = len(raw.split(".")[1]) if "." in raw else 0
            best = None
            for fact in pool:
                if not _matches(value, decimals, fact):
                    continue
                rank = support_rank(value, decimals, fact)
                if best is None or rank < best:
                    best = rank
                if best == 0:
                    break
            total += 1
            if best is None:
                unsupported += 1
            else:
                counts[best] += 1
                if best >= 2:
                    scaled_here = True
        if scaled_here:
            examples_with_scaled.append(example.get("id", "?"))

    return {
        "what_this_is": (
            "Every numeral in every committed narrative, counted by the closest way layer "
            "one supports it. This prices the loose part of the match; it is not a semantic "
            "error rate and must not be read as one."
        ),
        "narratives": len(examples),
        "numerals_total": total,
        "numerals_unsupported": unsupported,
        "supported_by_the_fact_as_stored": counts[0],
        "supported_by_the_fact_rounded": counts[1],
        "supported_only_by_a_percentage_reading": counts[2] + counts[3],
        "narratives_containing_a_percentage_only_numeral": len(examples_with_scaled),
        "which_narratives": examples_with_scaled,
        "limits": (
            "This counts how a numeral is supported, never whether it was bound to the right "
            "fact. Both loose tiers can hide a coincidence: rounding is a half-unit window, so "
            "a stored 1.5 carries a printed 2, and the percentage reading moves the decimal "
            "point. Layer one matches numerals and does not read sentences, so the noun-binding "
            "question stays unmeasured, and the one miss the console reaches was found by "
            "reading rather than by any check here."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit", metavar="PATH", help="write the audit to this results file")
    args = parser.parse_args()

    data = json.loads(NARRATIVES.read_text(encoding="utf-8"))
    report = audit(data.get("examples") or [])

    for key, value in report.items():
        if key in ("what_this_is", "limits", "which_narratives"):
            continue
        print(f"  {key}: {value}")
    print(f"  narratives with one: {', '.join(report['which_narratives']) or 'none'}")

    if args.emit:
        Path(args.emit).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
