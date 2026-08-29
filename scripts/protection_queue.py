#!/usr/bin/env python3
"""
protection_queue.py: restate the protection result in the unit a review queue runs in.

Why this exists
---------------
The protection exhibit reports two recall fractions at the top one percent of a ranking:
0.5989 for the backbone score added to the counting control, 0.3228 for that control alone.
A recall fraction does not survive being retold in a meeting. The same fact in operational
units does: at a fixed review capacity, the queue catches 566 of the fraud transactions
instead of 305.

Nothing here is a new measurement, and nothing here is an assumption. Every output is exact
arithmetic over four numbers already stored in results/protection.json, and the two recalls
turn out to be exactly 566/945 and 305/945, so the caught counts are the underlying integer
counts rather than rounded products. The source pointers are written into the output so the
page can open them and a reader can redo the multiplication.

The generator of results/protection.json needs torch and the pretrained checkpoint, which is a
cluster job. This runs on the committed file instead, which is why it is a separate script and
a separate results file rather than an extra block in that one.

Run: python3 scripts/protection_queue.py     writes results/protection_queue.json
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derivation_check import write_or_check

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "protection.json"
OUT = ROOT / "results" / "protection_queue.json"

TOP_FRACTION = Fraction(1, 100)  # the operating point the exhibit already reports

POINTERS = {
    "n_scored_rows": "/split/n_scored_rows",
    "n_positives": "/split/n_positives",
    "recall_model": "/scores/pll_plus_rarity/recall_at_top_0.01",
    "recall_control": "/scores/rarity_all_fields/recall_at_top_0.01",
}


def main() -> int:
    d = json.loads(SRC.read_text())
    rows = d["split"]["n_scored_rows"]
    positives = d["split"]["n_positives"]
    r_model = d["scores"]["pll_plus_rarity"]["recall_at_top_0.01"]
    r_control = d["scores"]["rarity_all_fields"]["recall_at_top_0.01"]

    queue = int(rows * TOP_FRACTION)
    caught_model = r_model * positives
    caught_control = r_control * positives
    # If these are not integers the stored recalls are not k/n and the restatement would be
    # inventing precision, so refuse rather than round.
    for name, v in (("model", caught_model), ("control", caught_control)):
        if abs(v - round(v)) > 1e-6:
            print(f"REFUSING: recall x positives for the {name} score is {v}, not an integer, "
                  f"so the caught count is not recoverable exactly from the stored fraction.")
            return 1
    caught_model, caught_control = round(caught_model), round(caught_control)

    payload = {
        "what_this_is": (
            "The protection result restated in the unit a review queue runs in. This file contains "
            "NO new measurement: every value is exact arithmetic over four numbers stored in "
            "results/protection.json, listed under derived_from. The two stored recalls are exactly "
            f"{caught_model}/{positives} and {caught_control}/{positives}, so the caught counts are "
            "the underlying integer counts and not rounded products."
        ),
        "derived_from": {"file": "results/protection.json", "pointers": POINTERS},
        "generated_by": "scripts/protection_queue.py",
        "labels": d.get("labels", []),
        "carries_the_same_caveats_as_the_source": (
            "The corpus is the synthetic IBM TabFormer benchmark, the positive rate is thinned by "
            "the split, and the labels are card-fraud labels rather than authorized-scam labels. "
            "Restating the result in queue units changes none of that."
        ),
        "operating_point": "top 1 percent of the ranking, the same operating point the exhibit reports",
        "n_scored_rows": rows,
        "review_queue_rows": queue,
        "n_positives": positives,
        "caught_with_backbone_score": caught_model,
        "caught_control_alone": caught_control,
        "additional_caught": caught_model - caught_control,
        "relative_increase": (caught_model - caught_control) / caught_control,
    }
    rc = write_or_check(OUT, payload)
    if rc == 0 and "--check" not in sys.argv[1:]:
        print(f"wrote {OUT.relative_to(ROOT)}: a {queue:,}-row queue catches {caught_model} "
              f"instead of {caught_control}, {caught_model - caught_control} more, "
              f"{payload['relative_increase']:.4f} relative.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
