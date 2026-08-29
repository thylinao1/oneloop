#!/usr/bin/env python3
"""
narratives_direction_audit.py: run the direction layer over the shipped narratives.

Why this exists
---------------
The generated-text gate had two layers: a deterministic numeric check, and a cross-examination
by the served model. Thirty of thirty narratives passed both. A third, deterministic layer built
afterwards fails five of them, and every numeral in those five is correct. What is wrong is the
direction: they say the corridor model outperformed the seasonal naive while their own facts say
the naive was ahead.

The cause was a fact bundle rather than a hallucination. make_inputs.py told the generator "MASE
below 1 beats the seasonal-naive baseline", and every reversed narrative has a model MASE below
one, so the model applied the rule it was handed. The cross-examination could not catch it: it is
the same model reading the same wrong premise. The false rule is fixed in make_inputs.py, so a
regeneration produces correct text, and this audit is what keeps the claim honest until then.

This is the same shape as the injection hole our own red team found in this gate and which the
entry already prints: found, diagnosed, gate hardened, re-measured, published. The number that
matters is not the pass rate before the layer existed, it is what the layer found.

Run: python3 scripts/narratives_direction_audit.py            writes results/narratives_direction.json
     python3 scripts/narratives_direction_audit.py --check    recompute and compare at 1e-6
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "narratives"))
from derivation_check import write_or_check  # noqa: E402
from directioncheck import COMPARISONS, direction_failures, is_checkable  # noqa: E402

SRC = ROOT / "results" / "narratives.json"
OUT = ROOT / "results" / "narratives_direction.json"


def main() -> int:
    d = json.loads(SRC.read_text())
    examples = d["examples"]

    checked, failures = [], []
    for e in examples:
        facts = e.get("input_facts") or {}
        text = e.get("narrative") or e.get("text") or ""
        if not is_checkable(facts):
            continue
        fails = direction_failures(text, facts)
        checked.append(e["id"])
        if fails:
            failures.append({
                "id": e["id"],
                "gate_verdict_before_this_layer": e.get("verdict"),
                "mase_model": facts.get("mase_model"),
                "mase_seasonal_naive": facts.get("mase_seasonal_naive"),
                "reason": fails[0],
            })

    payload = {
        "what_this_is": (
            "A third, deterministic layer on the generated-text gate, checking the DIRECTION of a "
            "comparison rather than the values in it. The two shipped layers pass every narrative "
            "here; this one does not, and every numeral in the narratives it fails is correct."
        ),
        "generated_by": "scripts/narratives_direction_audit.py",
        "layer_source": "scripts/narratives/directioncheck.py",
        "labels": d.get("labels", []),
        "why_the_earlier_layers_missed_it": (
            "Layer one proves every numeral matches a fact value, which these narratives do. Layer "
            "two is a cross-examination by the same served model that wrote the text, given the "
            "same fact bundle, and that bundle contained the false rule 'MASE below 1 beats the "
            "seasonal-naive baseline'. Every reversed narrative has a model MASE below one, so the "
            "model applied the rule it was handed and then confirmed its own application of it. A "
            "gate that checks whether numbers are right is not a gate that checks whether the "
            "sentence about them is right."
        ),
        "root_cause_fixed_in": "scripts/narratives/make_inputs.py, which now states that MASE is "
                               "an error measure, that lower is better, and that comparing either "
                               "number to 1 says nothing about which of the two won",
        "regeneration_status": (
            "NOT REGENERATED. Generation needs a self-hosted Qwen3-32B on a cluster GPU, which is "
            "a manual action, so the corrected fact bundle has not yet produced new text. The "
            "narratives in results/narratives.json are the ones that were generated under the "
            "false rule, and this file is what stops them being read as clean."
        ),
        "n_examples_total": len(examples),
        "n_checkable": len(checked),
        "n_unchecked": len(examples) - len(checked),
        "n_direction_failures": len(failures),
        "comparisons_declared": [c["label"] for c in COMPARISONS],
        "coverage_note": (
            "Only comparisons declared in directioncheck.py are checked. A bundle this layer does "
            "not understand is counted as UNCHECKED rather than passed, because a gate that "
            "silently passes what it cannot read is the failure this whole exercise is about."
        ),
        "failures": failures,
        # Keyed by example id so the replay console can bind a panel's direction verdict without
        # an array index. "fail" carries the reason; "pass" and "unchecked" are stated outright,
        # because a gate that silently passes what it cannot read is the failure this file is about.
        "by_id": {
            e["id"]: (
                {"direction": "fail",
                 "reason": next(f["reason"] for f in failures if f["id"] == e["id"])}
                if any(f["id"] == e["id"] for f in failures)
                else {"direction": "pass" if e["id"] in checked else "unchecked",
                      "reason": ("the declared comparison holds in the direction stated"
                                 if e["id"] in checked else
                                 "no declared comparison applies to this bundle, so this layer "
                                 "makes no claim about it")}
            )
            for e in examples
        },
    }
    rc = write_or_check(OUT, payload)
    if rc or "--check" in sys.argv[1:]:
        return rc
    print(f"checkable {len(checked)} of {len(examples)} | direction failures {len(failures)}")
    for f in failures:
        print(f"  {f['id']}: model {f['mase_model']} against naive {f['mase_seasonal_naive']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
