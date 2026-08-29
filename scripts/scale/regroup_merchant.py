"""regroup_merchant.py: merchant-view prep dir from the committed cardholder prep.

Reuses scripts/fm tokenization untouched: the SAME tokenized events are re-sorted
by (merchant, ts) and the sequence-index array ('user.npy') is replaced with the
merchant id, so fm's WindowDataset / user_segments / embed.py / transfer_eval.py
all group per MERCHANT, ordered in time. This is the partial two-sided
demonstration: entity = merchant throughout; the same cut_ts applies (identical
event set, identical timeline), vocab identical.

Disclosed caveat (recorded in meta): the transfer split is merchant-disjoint but
cardholders cross merchant boundaries.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ARRAYS = ("tokens", "user", "ts", "fraud", "mcc_class", "amount", "merchant")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-in", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, out = Path(args.prep_in), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = {name: np.load(src / f"{name}.npy") for name in ARRAYS}
    n = len(d["ts"])
    order = np.lexsort((d["ts"], d["merchant"]))  # primary: merchant, secondary: ts

    merch_sorted = d["merchant"][order]
    ts_sorted = d["ts"][order]
    # sanity: contiguous merchant groups, time-ordered within each group
    same_group = np.diff(merch_sorted) == 0
    assert np.all(np.diff(merch_sorted) >= 0), "merchant groups not contiguous"
    assert np.all(np.diff(ts_sorted)[same_group] >= 0), "time order broken within merchant"

    np.save(out / "tokens.npy", d["tokens"][order])
    np.save(out / "user.npy", merch_sorted.astype(np.int32))  # sequence index := merchant
    np.save(out / "ts.npy", ts_sorted.astype(np.int64))
    np.save(out / "fraud.npy", d["fraud"][order])
    np.save(out / "mcc_class.npy", d["mcc_class"][order])
    np.save(out / "amount.npy", d["amount"][order])
    np.save(out / "merchant.npy", merch_sorted.astype(np.int32))

    meta = json.loads((src / "meta.json").read_text())
    meta["axis"] = "merchant"
    meta["regrouped_from"] = str(src)
    meta["regroup_note"] = (
        "same tokenized events re-sorted by (merchant, ts); sequence entity = merchant; "
        "cut_ts/vocab identical to cardholder prep; caveat: cardholders cross merchant "
        "boundaries, so the merchant-disjoint transfer split is entity-disjoint in "
        "merchants only (partial two-sided demonstration)"
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.copy2(src / "merchant_keys.parquet", out / "merchant_keys.parquet")

    n_merch = int(merch_sorted.max()) + 1
    print(f"[regroup] {n} events re-grouped into {n_merch} merchant sequences -> {out}", flush=True)


if __name__ == "__main__":
    main()
