"""patch_membership_strings.py, SAFE-A: apply the reviewed prose corrections.

WHY THIS SCRIPT EXISTS, AND WHY IT IS NOT A HAND EDIT

  Every other safety file was corrected by editing its producer and re-running it, which
  is the honest route: the whole file regenerates and every number reproduces on the way
  past. SAFE-A cannot take that route here. `mia_eval.py --stage run` needs the prep
  arrays, the frozen checkpoint and a CUDA device, none of which exist on the laptop this
  correction was made on, and its `--check` is a cluster job pinned to the producing node.

  So the corrections land in two places at once and this script is the seam between them:

    1. `mia_eval.py` holds the corrected wording in pure helper functions that take plain
       values and touch no array: population_note, overlap_curve_note,
       matching_parameters, confound_evidence_note, DEVIATION_TENURE_MATCHED and
       DESIGNED_NOT_RUN. A future re-run of the producer emits exactly these strings.
    2. This script IMPORTS THOSE SAME FUNCTIONS and writes their output into the committed
       results file. It does not retype the wording, so the file and the producer cannot
       drift apart in the way a hand edit would let them.

WHAT IT IS ALLOWED TO TOUCH

  Strings only. It asserts, before and after, that the file's numeric leaf map is
  IDENTICAL: same paths, same values, same count. If a single numeric leaf appears,
  disappears or moves, it writes nothing and exits non-zero. The counts it needs
  (the sampled / contamination-gradient overlap) are derived from leaves the file already
  publishes, by |A and B| = |A| + |B| - |A or B|, and they go into a string.

  It is idempotent: running it twice produces the same bytes, so `--check` re-applies
  everything and asserts the file is already at the fixed point.

Usage:  python scripts/safety/patch_membership_strings.py [--check]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "fm"))

import mia_eval as M  # noqa: E402
from common import atomic_write_json  # noqa: E402

TARGET = REPO / "results" / "safety_membership.json"


def numeric_leaves(obj, prefix=""):
    """Byte-for-byte the rule protection_pll.numeric_leaves uses: bools are not numbers."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield prefix, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from numeric_leaves(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from numeric_leaves(v, f"{prefix}/{i}")


def reorder(d: dict, key: str, value, before: str) -> dict:
    """Insert key immediately before `before`, so the key order mirrors the producer.

    Drops any existing copy of `key` first, so a second run reproduces the first run's
    bytes exactly instead of leaving the old value pinned at its old position.
    """
    if before not in d:
        raise KeyError(f"anchor {before!r} not found")
    out = {}
    for k, v in d.items():
        if k == key:
            continue
        if k == before:
            out[key] = value
        out[k] = v
    return out


def run_parameters(d: dict) -> tuple[float, int, int]:
    """Recover the caliper, the pair floor and the non-member account count.

    Read back out of the committed file's own prose rather than assumed, then
    cross-checked against mia_eval's argparse defaults and against the asserted counts.
    scripts/safety/job_safe_a.sbatch passes neither --caliper nor --min-pairs, so the
    producing run took the defaults, and that is the receipt for the cross-check.
    """
    tenure = d["diagnostics_not_attacks"]["account_tenure"]
    reason = tenure["matching"].get("reason") or ""
    m = re.search(r"caliper of ([0-9.]+) dex", reason)
    n = re.search(r"below the floor of (\d+)", reason)
    if not (m and n):
        raise SystemExit("cannot recover caliper / pair floor from matching.reason")
    caliper, floor = float(m.group(1)), int(n.group(1))

    defaults = {a.dest: a.default for a in _producer_defaults()}
    if defaults["caliper"] != caliper or defaults["min_pairs"] != floor:
        raise SystemExit(f"file says caliper={caliper} floor={floor}, producer defaults say "
                         f"{defaults['caliper']} / {defaults['min_pairs']}; refusing to patch")

    acc = re.search(r"out of (\d+) non-member accounts available", tenure["overlap_curve_note"])
    if not acc:
        raise SystemExit("cannot recover the non-member account count")
    n_accounts = int(acc.group(1))
    asserted = d["membership_rule"]["asserted_counts"]["n_non_member_eligible_accounts"]
    if n_accounts != asserted:
        raise SystemExit(f"account count {n_accounts} disagrees with asserted {asserted}")
    return caliper, floor, n_accounts


def _producer_defaults():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caliper", type=float, default=0.2)
    ap.add_argument("--min-pairs", type=int, default=30)
    src = (HERE / "mia_eval.py").read_text()
    for flag, cast in (("--caliper", float), ("--min-pairs", int)):
        hit = re.search(rf'add_argument\("{flag}", type=\w+, default=([0-9.]+)', src)
        if not hit:
            raise SystemExit(f"cannot read the producer default for {flag}")
        for a in ap._actions:
            if a.option_strings and a.option_strings[0] == flag:
                a.default = cast(hit.group(1))
    return [a for a in ap._actions if a.option_strings]


def patch(d: dict) -> dict:
    caliper, floor, n_accounts = run_parameters(d)

    pop = d["population"]
    n_scored = pop["n_scored_rows"]
    n_sample = pop["n_member_rows_sampled"]
    n_diag = pop["n_member_rows_contamination_gradient_only"]
    n_non = pop["n_non_member_rows"]
    # The scored member set is union(member_sample, member_diag); see build_population.
    n_union = n_scored - n_non
    n_both = n_sample + n_diag - n_union
    sample_only, diag_only = n_sample - n_both, n_diag - n_both
    if min(n_both, sample_only, diag_only) < 0 or sample_only + n_both + diag_only + n_non != n_scored:
        raise SystemExit("the population decomposition does not close; refusing to patch")

    pop["note"] = M.population_note(n_scored, sample_only, n_both, diag_only, n_non)

    tenure = d["diagnostics_not_attacks"]["account_tenure"]
    tenure["overlap_curve_note"] = M.overlap_curve_note(n_accounts, caliper, floor)
    d["diagnostics_not_attacks"]["account_tenure"] = reorder(
        tenure, "matching_parameters", M.matching_parameters(caliper, floor), before="matching")

    matched_ran = bool(tenure["matching"].get("run"))
    d["headline_reading"]["what_the_evidence_bearing_on_that_confound_is"] = (
        M.confound_evidence_note(matched_ran))

    devs = d["deviations_from_preregistration"]
    hits = [i for i, s in enumerate(devs) if s.startswith("A TENURE-MATCHED STRATUM")]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one tenure-matched deviation, found {len(hits)}")
    devs[hits[0]] = M.DEVIATION_TENURE_MATCHED

    d = reorder(d, "designed_not_run", M.DESIGNED_NOT_RUN,
                before="deviations_from_preregistration")
    return reorder(d, "designed_not_run_how_to_read", M.DESIGNED_NOT_RUN_HOW_TO_READ,
                   before="designed_not_run")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="re-apply and assert the file is already at the fixed point")
    args = ap.parse_args()

    raw = TARGET.read_text()
    before = json.loads(raw)
    leaves_before = dict(numeric_leaves(before))

    after = patch(json.loads(raw))
    leaves_after = dict(numeric_leaves(after))

    if leaves_before != leaves_after:
        gone = sorted(set(leaves_before) - set(leaves_after))
        new = sorted(set(leaves_after) - set(leaves_before))
        moved = [k for k in set(leaves_before) & set(leaves_after)
                 if leaves_before[k] != leaves_after[k]]
        print(f"REFUSED: numeric leaves changed. removed={gone[:5]} added={new[:5]} "
              f"moved={moved[:5]}", file=sys.stderr)
        return 2
    print(f"numeric leaves: {len(leaves_before)} before, {len(leaves_after)} after, identical map")

    new_text = json.dumps(after, indent=1)
    if args.check:
        ok = new_text == raw
        print("CHECK OK: file is already at the fixed point" if ok else
              "CHECK FAILED: file is not at the fixed point")
        return 0 if ok else 1

    if new_text == raw:
        print("no change needed")
        return 0
    atomic_write_json(TARGET, after)
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
