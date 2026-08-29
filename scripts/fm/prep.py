"""prep.py: TabFormer CSV -> tokenized arrays for the One Loop backbone.

Leakage hard-fails implemented HERE:
  (a) 'Is Fraud?' excluded from model inputs/vocab (label only).
  (d) User/Card excluded from the vocab entirely (sequence index only).
Vocabularies + amount quantile edges are built on PRE-cut rows only
(post-cut unseen values map to UNK), so nothing post-cut shapes the vocab.

Usage (smoke): prep.py --csv sample.csv --out prep/ --max-rows 2000000 --sample-stride 40
Usage (full):  prep.py --csv card_transaction.v1.csv --out prep/
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from common import (
    FIELDS, FORBIDDEN_COLUMNS, N_SPECIAL, UNK_ID,
    TABFORMER_SHA256, TABFORMER_URL, atomic_write_json, seed_everything, versions_dict,
)

CITY_TOP_K = 1000
AMOUNT_BUCKETS = 100
MCC_TOP_K = 30  # next-MCC task: top-30 classes + 1 'other'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=0, help="read only first N csv rows (0=all)")
    ap.add_argument("--sample-stride", type=int, default=1, help="keep every k-th row (smoke: widen user coverage)")
    ap.add_argument("--post-frac", type=float, default=0.18, help="fraction of timeline after cut T")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-if-done", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    if args.skip_if_done and (out / "meta.json").exists():
        print(f"[prep] {out}/meta.json exists; skipping (--skip-if-done)")
        return
    seed_everything(args.seed)
    out.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(
        args.csv,
        n_rows=args.max_rows or None,
        schema_overrides={
            "User": pl.Int64, "Card": pl.Int64, "Year": pl.Int32, "Month": pl.Int32,
            "Day": pl.Int32, "Time": pl.Utf8, "Amount": pl.Utf8, "Use Chip": pl.Utf8,
            "Merchant Name": pl.Utf8, "Merchant City": pl.Utf8, "Merchant State": pl.Utf8,
            "Zip": pl.Utf8, "MCC": pl.Int64, "Errors?": pl.Utf8, "Is Fraud?": pl.Utf8,
        },
    )
    if args.sample_stride > 1:
        df = df.gather_every(args.sample_stride)
    n_raw = df.height
    print(f"[prep] rows read: {n_raw}")

    df = df.with_columns(
        pl.col("Time").str.slice(0, 2).cast(pl.Int32, strict=False).alias("hour"),
        pl.col("Time").str.slice(3, 2).cast(pl.Int32, strict=False).alias("minute"),
        pl.col("Amount").str.replace_all(r"[$,]", "").cast(pl.Float64, strict=False).alias("amount"),
    ).with_columns(
        pl.datetime("Year", "Month", "Day", "hour", "minute").alias("dtm")
    )
    n_bad = df.filter(pl.col("dtm").is_null() | pl.col("amount").is_null()).height
    if n_bad:
        print(f"[prep] dropping {n_bad} rows with unparseable date/amount")
        df = df.filter(pl.col("dtm").is_not_null() & pl.col("amount").is_not_null())
    df = df.with_columns((pl.col("dtm").cast(pl.Int64) // 1_000_000).alias("ts"))
    df = df.sort(["User", "ts"], maintain_order=True)

    ts = df["ts"].to_numpy()
    cut_ts = int(np.quantile(ts, 1.0 - args.post_frac))
    corpus_cut_date = dt.datetime.utcfromtimestamp(cut_ts).strftime("%Y-%m-%dT%H:%M:%SZ")
    pre = ts < cut_ts
    print(f"[prep] cut T = {corpus_cut_date} | pre-T rows {int(pre.sum())} | post-T rows {int((~pre).sum())}")

    # ---- categorical string fields (raw values) ----
    df = df.with_columns(
        pl.col("Year").cast(pl.Utf8).alias("year"),
        pl.col("Month").cast(pl.Utf8).alias("month"),
        pl.col("Day").cast(pl.Utf8).alias("day"),
        pl.col("hour").cast(pl.Utf8).alias("hour_s"),
        pl.col("Use Chip").fill_null("NA").alias("use_chip"),
        pl.col("MCC").cast(pl.Utf8).fill_null("NA").alias("mcc"),
        pl.col("Merchant City").fill_null("NA").alias("city"),
        pl.col("Merchant State").fill_null("NA").alias("state"),
        pl.col("Errors?").fill_null("none").alias("errors"),
    )

    # amount quantile buckets; edges from PRE-cut rows only
    amount = df["amount"].to_numpy().astype(np.float32)
    edges = np.quantile(amount[pre], np.linspace(0, 1, AMOUNT_BUCKETS + 1)[1:-1])
    amount_q = np.searchsorted(edges, amount).astype(np.int64)  # 0..99

    raw_cols = {
        "year": "year", "month": "month", "day": "day", "hour": "hour_s",
        "use_chip": "use_chip", "mcc": "mcc", "city": "city", "state": "state",
        "errors": "errors",
    }
    for col in FORBIDDEN_COLUMNS:
        assert col not in raw_cols.values() and col not in FIELDS, f"LEAKAGE: {col} would enter vocab"

    pre_df = df.with_row_index("_i").filter(pl.Series(pre))
    vocabs: dict[str, dict[str, int]] = {}
    token_cols: dict[str, np.ndarray] = {}
    for field, col in raw_cols.items():
        vc = pre_df[col].value_counts(sort=True)
        vals = vc[col].to_list()
        if field == "city":
            vals = vals[:CITY_TOP_K]
        mapping = {v: N_SPECIAL + i for i, v in enumerate(vals)}
        vocabs[field] = mapping
        token_cols[field] = (
            df[col].replace_strict(mapping, default=UNK_ID, return_dtype=pl.Int32).to_numpy()
        )
    token_cols["amount_q"] = (N_SPECIAL + amount_q).astype(np.int32)
    vocab_sizes = {f: (N_SPECIAL + len(vocabs[f])) if f != "amount_q" else (N_SPECIAL + AMOUNT_BUCKETS)
                   for f in FIELDS}

    tokens = np.stack([token_cols[f] for f in FIELDS], axis=1)
    assert tokens.max() < 32000, "vocab too large for int16"
    tokens = tokens.astype(np.int16)

    # ---- labels / auxiliary arrays (NOT model inputs) ----
    fraud = (df["Is Fraud?"] == "Yes").to_numpy().astype(np.int8)
    user = df["User"].to_numpy().astype(np.int32)

    # next-MCC classes from PRE-cut frequency; top-K + other; unseen -> other
    mcc_vc = pre_df["mcc"].value_counts(sort=True)
    mcc_vals = mcc_vc["mcc"].to_list()[:MCC_TOP_K]
    mcc_map = {v: i for i, v in enumerate(mcc_vals)}
    mcc_class = df["mcc"].replace_strict(mcc_map, default=MCC_TOP_K, return_dtype=pl.Int32).to_numpy().astype(np.int16)

    # merchant key = Merchant Name + City (pooling index for embed.py; never in vocab)
    df = df.with_columns(
        (pl.col("Merchant Name").fill_null("NA") + "||" + pl.col("city")).alias("merch_key")
    )
    merch_cat = df["merch_key"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int32)
    merch_keys = (
        df.select("merch_key", "Merchant Name", "city")
        .with_columns(pl.Series("merchant_id", merch_cat))
        .unique(subset=["merchant_id"])
        .sort("merchant_id")
        .select("merchant_id", pl.col("Merchant Name").alias("merchant_name"), pl.col("city").alias("merchant_city"))
    )

    np.save(out / "tokens.npy", tokens)
    np.save(out / "user.npy", user)
    np.save(out / "ts.npy", ts.astype(np.int64))
    np.save(out / "fraud.npy", fraud)
    np.save(out / "mcc_class.npy", mcc_class)
    np.save(out / "amount.npy", amount)
    np.save(out / "merchant.npy", merch_cat)
    merch_keys.write_parquet(out / "merchant_keys.parquet")

    meta = {
        "fields": FIELDS,
        "vocab_sizes": [int(vocab_sizes[f]) for f in FIELDS],
        "amount_edges": [float(e) for e in edges],
        "cut_ts": cut_ts,
        "corpus_cut_date": corpus_cut_date,
        "post_frac": args.post_frac,
        "n_rows": int(len(user)),
        "n_users": int(len(np.unique(user))),
        "n_merchants": int(merch_keys.height),
        "mcc_top_k": MCC_TOP_K,
        "n_mcc_classes": MCC_TOP_K + 1,
        "sample_stride": args.sample_stride,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "versions": versions_dict(),
        "data_sources": [{"name": "IBM TabFormer card_transaction.v1.csv (synthetic)",
                          "url": TABFORMER_URL, "sha256": TABFORMER_SHA256}],
        "leakage": {
            "label_excluded": True,   # 'Is Fraud?' never tokenized (asserted above)
            "ids_excluded": True,     # User/Card never tokenized (asserted above)
            "vocab_built_pre_cut_only": True,
        },
    }
    atomic_write_json(out / "meta.json", meta)
    print(f"[prep] done: {len(user)} rows, {meta['n_users']} users, {meta['n_merchants']} merchants -> {out}")


if __name__ == "__main__":
    main()
