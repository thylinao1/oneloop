"""make_scaling_entry.py: reshape one fm transfer_eval output into a CONTRACT §2
scaling entry (the backbone_transfer.json scaling extension).

Emits scale/<tag>/transfer.json: common envelope + {rows, params_m, axis, seed,
tasks} with the SAME task shapes as backbone_transfer.json's tasks key. Stage-2
folds these files into results/backbone_transfer.json's "scaling" list.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def downsample(curve: list, max_points: int = 400) -> list:
    if len(curve) <= max_points:
        return curve
    step = len(curve) / max_points
    idx = sorted({min(len(curve) - 1, int(i * step)) for i in range(max_points)} | {len(curve) - 1})
    return [curve[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", required=True, help="raw transfer_eval.py output json")
    ap.add_argument("--prep-meta", required=True)
    ap.add_argument("--axis", required=True, choices=["cardholder", "merchant"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t = json.loads(Path(args.transfer).read_text())
    m = json.loads(Path(args.prep_meta).read_text())

    entry = {
        # ---- common envelope (CONTRACT §2) ----
        "seed": t["seed"],
        "versions": t["versions"],
        "generated_by": "scripts/scale/make_scaling_entry.py (reshapes scripts/fm/transfer_eval.py output) --check-able",
        "data_sources": t["data_sources"],
        "labels": t["labels"],
        # ---- scaling-entry keys ----
        "tag": args.tag,
        "rows": int(m["n_rows"]),
        "params_m": t["pretrain"]["params_m"],
        "axis": args.axis,
        "tasks": t["tasks"],
        # ---- provenance kept for the record ----
        "pretrain": {**t["pretrain"], "loss_curve": downsample(t["pretrain"]["loss_curve"])},
        "leakage_checks": t["leakage_checks"],
        "corpus_cut_date": m["corpus_cut_date"],
        "n_users": int(m.get("n_users", 0)),
        "n_merchants": int(m.get("n_merchants", 0)),
        "split": t.get("split"),
        "bootstrap": t.get("bootstrap"),
        "note": args.note,
    }
    tmp = Path(args.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, indent=1))
    tmp.replace(args.out)
    fr = entry["tasks"]["fraud"]
    print(
        f"[entry] {args.tag}: rows={entry['rows']} params_m={entry['params_m']} axis={entry['axis']} "
        f"seed={entry['seed']} fraud dAUC {fr['with_emb_auc'] - fr['baseline_auc']:+.4f} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
