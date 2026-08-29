"""Re-run ONLY the deterministic numeric faithfulness layer against the stored
narratives + stored facts in results/narratives.json (no regeneration, no GPU,
no network). CPU, stdlib, 1e-6 tolerance. Exit 0 iff every stored numeric
verdict reproduces exactly.

Usage: python3 scripts/narratives/check.py --check
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from faithcheck import unsupported_numerals  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", required=True)
    ap.add_argument("--file", default=str(ROOT / "results" / "narratives.json"))
    args = ap.parse_args()

    d = json.loads(Path(args.file).read_text())
    mismatches = 0
    for ex in d["examples"]:
        got = unsupported_numerals(ex["narrative"], ex["input_facts"])
        if got != ex["numeric_unsupported"]:
            mismatches += 1
            print(f"MISMATCH {ex['id']}: stored={ex['numeric_unsupported']} recomputed={got}")
        # verdict consistency: numeric layer of the stored verdict must agree
        stored_numeric_fail = bool(ex["numeric_unsupported"])
        if bool(got) != stored_numeric_fail:
            mismatches += 1
            print(f"VERDICT-INCONSISTENT {ex['id']}")
    n = len(d["examples"])
    strict = d["faithfulness"]["n_pass_strict"]
    recomputed_strict = sum(
        1 for ex in d["examples"]
        if not ex["numeric_unsupported"] and not ex["cross_exam_flags"])
    if recomputed_strict != strict:
        mismatches += 1
        print(f"STRICT-COUNT MISMATCH stored={strict} recomputed={recomputed_strict}")
    if mismatches:
        print(f"CHECK FAILED: {mismatches} mismatches over {n} examples")
        sys.exit(1)
    print(f"CHECK OK: numeric layer reproduces for all {n} examples "
          f"(strict pass {strict}/{n})")
    sys.exit(0)


if __name__ == "__main__":
    main()
