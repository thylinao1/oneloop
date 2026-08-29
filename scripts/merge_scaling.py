#!/usr/bin/env python3
"""merge_scaling.py: merge the scaling-curve results (stage 2).

Folds the scaling-curve transfer results (results/scale/*.json, produced by
scripts/scale/make_scaling_entry.py) into results/backbone_transfer.json's
optional "scaling" key (the scaling-curve schema):

    "scaling": [{"rows", "params_m", "axis", "seed", "tasks"}...], "axes_note": str

and ALSO emits "scaling_by_tag": {tag -> the same entry object}, so prose slots
can address a run by name instead of by list position. The list is sorted by
(axis, rows, seed, tag) for the table, which means a positional pointer like
scaling.0 CHANGES MEANING when a new run lands (this bit us once: scaling.0
moved from the 10M point to the 3M point when scale-3m merged, silently
swapping the hero fraud numbers). Copy slots MUST use scaling_by_tag.<tag>.

plus a SYNTHESIZED entry for the main 24M cardholder run, built from the
backbone file's own top-level `tasks` + `pretrain` metadata (its corpus row
count is taken from the scale entry that pretrained on the same full corpus,
matched by corpus_cut_date).

SAFELY RE-RUNNABLE: the scaling list and axes_note are rebuilt from whatever
entry files exist under results/scale/ on every run, so the still-queued
scale-3m and merchant-axis points merge in automatically when their cluster
jobs land and this script is re-run. Everything else in backbone_transfer.json
is preserved byte-identically (guarded: the script refuses to write if it
cannot reproduce the current file bytes from the parsed JSON).

Exit codes: 0 ok · 2 shape mismatch or fidelity guard tripped (nothing written).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKBONE = ROOT / "results" / "backbone_transfer.json"
SCALE_DIR = ROOT / "results" / "scale"

AXES = {"cardholder", "merchant"}
SYNTH_TAG = "backbone-24m-main"

# Page-visible descriptions per known tag (axes_note is judge-facing prose:
# no em/en dashes, no comma-grouped numerals; verify_numbers substring rule).
TAG_DESC = {
    "scale-3m": "a 3M row earliest-window subset (seed 7)",
    "scale-10m": "a 10M row earliest-window subset (seed 7)",
    SYNTH_TAG: "the full 24.4M row corpus (seed 7, the main run above)",
    "scale-24m-seed2": "a seed 1337 repeat of the full 24.4M row corpus as a seed stability check",
    "merchant-axis": "the same events regrouped into merchant sequences (merchant axis, seed 7)",
}
# Scaling points that were queued and may not have landed yet.
EXPECTED = {
    "scale-3m": "a 3M row subset",
    "merchant-axis": "a merchant-axis regrouping of the same events",
}


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_entry_shape(name: str, d: dict) -> list[str]:
    """Verify one scale entry against the CONTRACT §2 scaling-entry shape."""
    problems = []
    if not (isinstance(d.get("rows"), int) and d["rows"] > 0):
        problems.append(f"{name}: 'rows' missing or not a positive int ({d.get('rows')!r})")
    if not is_num(d.get("params_m")):
        problems.append(f"{name}: 'params_m' missing or non-numeric ({d.get('params_m')!r})")
    if d.get("axis") not in AXES:
        problems.append(f"{name}: 'axis' not in {sorted(AXES)} ({d.get('axis')!r})")
    if not isinstance(d.get("seed"), int):
        problems.append(f"{name}: 'seed' missing or not an int ({d.get('seed')!r})")
    tasks = d.get("tasks")
    if not (isinstance(tasks, dict) and tasks):
        problems.append(f"{name}: 'tasks' missing or empty")
    else:
        for tname, t in tasks.items():
            if not isinstance(t, dict):
                problems.append(f"{name}: tasks.{tname} is not an object")
        nm = tasks.get("next_mcc")
        if isinstance(nm, dict) and not isinstance(nm.get("delta_top1_ci"), list):
            # non-fatal for the renderer (cell shows a dash) but worth flagging
            problems.append(f"{name}: tasks.next_mcc lacks delta_top1_ci (renderer will dash the CI cell)")
    return problems


def to_scaling_entry(tag: str, d: dict, note: str) -> dict:
    """CONTRACT §2 scaling entry + tag/note provenance."""
    return {
        "tag": tag,
        "rows": d["rows"],
        "params_m": d["params_m"],
        "axis": d["axis"],
        "seed": d["seed"],
        "tasks": d["tasks"],
        "note": note,
    }


def describe(entry: dict) -> str:
    """Fallback axes_note description for a tag we did not anticipate."""
    if entry["tag"] in TAG_DESC:
        return TAG_DESC[entry["tag"]]
    return (f"a {entry['rows'] / 1e6:.1f}M row run on the {entry['axis']} axis "
            f"(seed {entry['seed']})")


def main() -> int:
    raw = BACKBONE.read_text(encoding="utf-8")
    doc = json.loads(raw)

    # Byte-fidelity guard: everything except the keys we own must survive
    # untouched. The file was written with json indent=1; refuse to proceed
    # if re-serialization cannot reproduce the current bytes exactly.
    if json.dumps(doc, indent=1) != raw:
        print("ABORT: backbone_transfer.json does not round-trip at indent=1; "
              "byte-identical preservation impossible; investigate before merging.",
              file=sys.stderr)
        return 2

    base = {k: v for k, v in doc.items()
            if k not in ("scaling", "scaling_by_tag", "axes_note")}

    # ---- collect + verify the landed scale entries ----------------------
    problems: list[str] = []
    entries: list[dict] = []
    scale_files = sorted(SCALE_DIR.glob("*.json")) if SCALE_DIR.is_dir() else []
    for path in scale_files:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            problems.append(f"{path.name}: unreadable ({e.__class__.__name__}: {e})")
            continue
        shape = check_entry_shape(path.name, d)
        fatal = [p for p in shape if "renderer will dash" not in p]
        for p in shape:
            print(("MISMATCH: " if p in fatal else "NOTE: ") + p, file=sys.stderr)
        if fatal:
            problems.extend(fatal)
            continue
        tag = d.get("tag") or path.stem
        entries.append(to_scaling_entry(tag, d, str(d.get("note", ""))))
        print(f"[merge] {path.name}: rows={d['rows']:,} params_m={d['params_m']} "
              f"axis={d['axis']} seed={d['seed']} OK")

    if problems:
        print(f"ABORT: {len(problems)} shape problem(s) in results/scale/; nothing written.",
              file=sys.stderr)
        return 2

    # ---- synthesize the main 24M cardholder point -----------------------
    # From the backbone file's own top-level tasks + pretrain metadata; its
    # corpus row count comes from the scale entry pretrained on the same full
    # corpus (identical corpus_cut_date, cardholder axis).
    cut = doc.get("pretrain", {}).get("corpus_cut_date")
    full_rows = None
    for path in scale_files:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("axis") == "cardholder" and d.get("corpus_cut_date") == cut \
                and isinstance(d.get("rows"), int):
            full_rows = d["rows"]
            break
    if full_rows is None:
        print("ABORT: cannot size the main-run corpus; no cardholder scale entry shares "
              f"the backbone corpus_cut_date {cut!r}. Nothing written.", file=sys.stderr)
        return 2
    if not (isinstance(doc.get("tasks"), dict) and doc["tasks"]
            and is_num(doc.get("pretrain", {}).get("params_m"))
            and isinstance(doc.get("seed"), int)):
        print("ABORT: backbone_transfer.json lacks top-level tasks/pretrain.params_m/seed "
              "needed to synthesize the main-run point.", file=sys.stderr)
        return 2
    entries.append({
        "tag": SYNTH_TAG,
        "rows": full_rows,
        "params_m": doc["pretrain"]["params_m"],
        "axis": "cardholder",
        "seed": doc["seed"],
        "tasks": doc["tasks"],
        "note": ("synthesized by scripts/merge_scaling.py from this file's own top-level "
                 "tasks + pretrain metadata (the main full-corpus cardholder run); rows "
                 "taken from the scale entry sharing corpus_cut_date " + str(cut)),
    })

    entries.sort(key=lambda e: (e["axis"], e["rows"], e["seed"], e["tag"]))

    # ---- axes_note (page-visible prose; the page style uses no em or en dashes) -----
    present_tags = {e["tag"] for e in entries}
    descs = "; ".join(describe(e) for e in entries)
    pending = [v for k, v in EXPECTED.items() if k not in present_tags]
    note = ("Each row is one full pretrain plus the same leakage-hardened transfer "
            "evaluation, on the axis shown, with the places the runs are not identical "
            "named in the card above. Points: " + descs + ".")
    if pending:
        note += (" Still queued at submission time, so absent from this table: "
                 + ", ".join(pending) + ".")
    note += " Deltas are reported as obtained."

    # ---- tag-addressed map (index-pointer seam killer) ------------------
    # Same entry objects, keyed by tag. Duplicate tags are a hard error: a
    # tag has to name exactly one run for the map to be trustworthy.
    by_tag: dict[str, dict] = {}
    for e in entries:
        if e["tag"] in by_tag:
            print(f"ABORT: duplicate scaling tag {e['tag']!r}; a tag must name exactly one run. "
                  "Nothing written.", file=sys.stderr)
            return 2
        by_tag[e["tag"]] = e

    merged = dict(base)
    merged["scaling"] = entries
    merged["scaling_by_tag"] = by_tag
    merged["axes_note"] = note

    BACKBONE.write_text(json.dumps(merged, indent=1), encoding="utf-8")

    print(f"[merge] wrote {BACKBONE.relative_to(ROOT)} with {len(entries)} scaling entries "
          f"and a scaling_by_tag map over {sorted(by_tag)}:")
    for e in entries:
        nm = e["tasks"].get("next_mcc", {})
        d1 = (nm.get("with_emb_top1", 0) - nm.get("baseline_top1", 0)
              if is_num(nm.get("with_emb_top1")) and is_num(nm.get("baseline_top1")) else None)
        ci = nm.get("delta_top1_ci")
        print(f"  {e['tag']:<18} rows={e['rows']:>10,} axis={e['axis']:<10} seed={e['seed']:<5}"
              + (f" next-MCC dtop1 {d1:+.4f} CI [{ci[0]:.4f}, {ci[1]:.4f}]"
                 if d1 is not None and isinstance(ci, list) and len(ci) == 2 else ""))
    if pending:
        print(f"[merge] pending (re-run to add): {', '.join(k for k in EXPECTED if k not in present_tags)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
