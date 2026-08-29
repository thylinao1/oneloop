"""ladder_readback.py: thin driver around the ladder's OWN --check read-back.

Why this exists: scripts/fm/ladder_eval.py's compare() prints a pass/fail verdict and the
count of numeric leaves, but not the largest observed difference. The read-back owed on
results/ladder.json needs that number recorded, so this driver:

  1. imports ladder_eval UNMODIFIED (the module file is not touched; its sha256 is
     printed by the job script on both sides of the run),
  2. wraps ladder_eval.compare so the ladder's own comparator still returns the verdict
     and still prints its own line, and
  3. additionally measures every numeric leaf's absolute difference and writes a report.

The verdict and the exit code come from ladder_eval.compare, not from this file.
Usage is identical to ladder_eval.py: pass the same arguments plus --check, and set
LADDER_READBACK_REPORT to the report path.
"""

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ladder_eval  # noqa: E402  (path must be set first)

_ORIG_COMPARE = ladder_eval.compare
_REPORT_PATH = os.environ.get("LADDER_READBACK_REPORT", "")


def _leaves(obj):
    return {
        k: v
        for k, v in ladder_eval.numeric_leaves(obj)
        if not k.startswith(ladder_eval.SKIP_CHECK_PREFIXES)
    }


def compare_with_maxdiff(fresh, stored, tol):
    """Call the ladder's own comparator, then record how far apart the leaves actually are."""
    recomputed = _leaves(fresh)
    committed = _leaves(stored)

    shared = sorted(set(recomputed) & set(committed))
    diffs = {k: abs(recomputed[k] - committed[k]) for k in shared}
    ranked = sorted(diffs.items(), key=lambda kv: kv[1], reverse=True)

    verdict = _ORIG_COMPARE(fresh, stored, tol)

    report = {
        "tolerance": tol,
        "n_leaves_recomputed": len(recomputed),
        "n_leaves_committed": len(committed),
        "n_leaves_compared": len(shared),
        "only_in_recomputed": sorted(set(recomputed) - set(committed)),
        "only_in_committed": sorted(set(committed) - set(recomputed)),
        "max_abs_diff": ranked[0][1] if ranked else None,
        "max_abs_diff_leaf": ranked[0][0] if ranked else None,
        "n_leaves_exactly_equal": sum(1 for _, d in ranked if d == 0.0),
        "largest_ten": [
            {"leaf": k, "abs_diff": d, "recomputed": recomputed[k], "committed": committed[k]}
            for k, d in ranked[:10]
        ],
        "ladder_eval_verdict_rc": verdict,
    }

    if ranked:
        line = (
            f"READBACK: compared {report['n_leaves_compared']} numeric leaves, "
            f"{report['n_leaves_exactly_equal']} exactly equal, "
            f"max abs diff {report['max_abs_diff']:.3e} at {report['max_abs_diff_leaf']}"
        )
    else:
        line = "READBACK: no comparable numeric leaves"
    print(line, flush=True)
    if report["only_in_recomputed"] or report["only_in_committed"]:
        print(
            f"READBACK: leaf-set mismatch, {len(report['only_in_recomputed'])} only recomputed, "
            f"{len(report['only_in_committed'])} only committed",
            flush=True,
        )
    if _REPORT_PATH:
        pathlib.Path(_REPORT_PATH).write_text(json.dumps(report, indent=1, sort_keys=True))
        print(f"READBACK: report written to {_REPORT_PATH}", flush=True)

    return verdict


ladder_eval.compare = compare_with_maxdiff

if __name__ == "__main__":
    raise SystemExit(ladder_eval.main())
