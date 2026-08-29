"""prep_cj.py - Complete Journey CSV/parquet -> tokenized arrays for the One Loop backbone.

Real corpus counterpart of prep.py (dunnhumby "The Complete Journey": 2,595,732
transactions, ~2,500 households, 711 days). Output is byte-compatible with what
pretrain.py consumes: the same seven array files plus meta.json with the same keys,
plus a `cj` provenance block. Run pretrain/embed through cj_run.py (which installs
CJ_FIELDS into common before the unmodified scripts execute).

Leakage hard-fails implemented HERE:
  (a) COUPON_DISC / COUPON_MATCH_DISC excluded from model inputs/vocab. They are the
      coupon-redemption outcome family that cj_offer_head.py predicts; letting the
      backbone read them would hand the offers demonstration its own answer.
      RETAIL_DISC stays IN: it is a shelf-price markdown set by the retailer before the
      purchase and observable on the receipt at transaction time, so it is a legitimate
      input in the way 'Is Fraud?' never was. The line is: fields the cardholder-side
      process observes at transaction time are inputs; fields that ARE the downstream
      outcome (or its family) are labels.
  (d) household_key excluded from the vocab entirely (sequence index only).
      STORE_ID excluded from the vocab entirely: prep.py keeps Merchant Name out of the
      vocab and represents merchants by pooled encodings, and the store head's
      embeddings-vs-counts question dies if the backbone can memorize store identity
      through a vocab id. STORE_ID is the pooling/join key (merchant.npy +
      merchant_keys.parquet), exactly the role prep.py gives its merchant key.
      BASKET_ID and PRODUCT_ID are identifiers, never tokenized; products enter as
      DEPARTMENT / COMMODITY_DESC / BRAND via the product.csv join.
Vocabularies + quantile edges are built on PRE-cut rows only (post-cut unseen values map
to UNK), so nothing post-cut shapes the vocab.

Time on this corpus: DAY is 1..711 with no calendar anchor, TRANS_TIME is an HHMM
integer. ts = (DAY-1)*86400 + HH*3600 + MM*60 (pseudo-epoch seconds). The corpus cut is
a DAY threshold: cut_day = quantile of DAY at (1 - post_frac), cut_ts = (cut_day-1)*86400,
so ts < cut_ts is exactly DAY < cut_day. dow = (DAY-1) % 7 is a 7-day cycle position,
not a named weekday (the guide does not anchor day 1 to a weekday).

Usage (smoke): prep_cj.py --txn transaction_data.parquet --product product.parquet \
               --out prep/ --max-rows 400000 --sample-stride 4 --skip-sha
Usage (full):  prep_cj.py --txn transaction_data.parquet --product product.parquet --out prep/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from common import N_SPECIAL, UNK_ID, atomic_write_json, seed_everything, versions_dict
from cj_common import (
    AMOUNT_BUCKETS, CJ_FIELDS, CJ_FORBIDDEN_COLUMNS, CJ_SOURCE_URL, CJ_TERMS,
    CJ_ZIP_SHA256, COMMODITY_TOP_CLASSES, COMMODITY_TOP_K, QUANTITY_MAX,
    RETAIL_DISC_BUCKETS, SECONDS_PER_DAY, WEEK_BUCKET, sha256_file,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txn", required=True, help="transaction_data.csv or .parquet")
    ap.add_argument("--product", required=True, help="product.csv or .parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=0, help="read only first N rows (0=all)")
    ap.add_argument("--sample-stride", type=int, default=1,
                    help="keep every k-th row (smoke: widen household coverage)")
    ap.add_argument("--post-frac", type=float, default=0.18,
                    help="fraction of the timeline after cut T (same default as prep.py)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-if-done", action="store_true")
    ap.add_argument("--skip-sha", action="store_true",
                    help="smoke only: skip hashing the input files")
    return ap.parse_args()


def read_table(path: str, n_rows: int) -> pl.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pl.read_parquet(p, n_rows=n_rows or None)
    return pl.read_csv(p, n_rows=n_rows or None, infer_schema_length=10000)


def clip_small_int(col: pl.Expr, hi: int) -> pl.Expr:
    """Small-int categorical: negatives -> 'neg', above hi -> '{hi+1}plus', else str."""
    return (
        pl.when(col < 0).then(pl.lit("neg"))
        .when(col > hi).then(pl.lit(f"{hi + 1}plus"))
        .otherwise(col.cast(pl.Utf8))
        .fill_null("NA")
    )


def quantile_bucket(values: np.ndarray, pre: np.ndarray, n_buckets: int):
    """Bucket edges from PRE-cut rows only; ties collapse buckets, which is fine."""
    edges = np.quantile(values[pre], np.linspace(0, 1, n_buckets + 1)[1:-1])
    return np.searchsorted(edges, values).astype(np.int64), edges


def main():
    args = parse_args()
    out = Path(args.out)
    if args.skip_if_done and (out / "meta.json").exists():
        print(f"[prep_cj] {out}/meta.json exists - skipping (--skip-if-done)")
        return
    seed_everything(args.seed)
    out.mkdir(parents=True, exist_ok=True)

    df = read_table(args.txn, args.max_rows)
    n_read = df.height
    missing = [c for c in ("household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY",
                           "SALES_VALUE", "STORE_ID", "RETAIL_DISC", "TRANS_TIME",
                           "WEEK_NO", "COUPON_DISC", "COUPON_MATCH_DISC")
               if c not in df.columns]
    assert not missing, f"transaction_data missing columns: {missing}"
    if args.sample_stride > 1:
        df = df.gather_every(args.sample_stride)
    print(f"[prep_cj] transaction rows read: {n_read}, kept after stride: {df.height}")

    prod = read_table(args.product, 0).select(
        "PRODUCT_ID",
        pl.col("DEPARTMENT").cast(pl.Utf8).fill_null("NA"),
        pl.col("COMMODITY_DESC").cast(pl.Utf8).fill_null("NA"),
        pl.col("BRAND").cast(pl.Utf8).fill_null("NA"),
    ).unique(subset=["PRODUCT_ID"], keep="first")
    df = df.join(prod, on="PRODUCT_ID", how="left")
    n_no_product = df.filter(pl.col("COMMODITY_DESC").is_null()).height
    df = df.with_columns(
        pl.col("DEPARTMENT").fill_null("NA"),
        pl.col("COMMODITY_DESC").fill_null("NA"),
        pl.col("BRAND").fill_null("NA"),
    )
    if n_no_product:
        print(f"[prep_cj] {n_no_product} rows have no product.csv match (kept, fields 'NA')")

    # ---- time: pseudo-epoch seconds from DAY + TRANS_TIME (HHMM int) ----
    df = df.with_columns(
        (pl.col("TRANS_TIME") // 100).clip(0, 23).alias("hour_i"),
        (pl.col("TRANS_TIME") % 100).clip(0, 59).alias("minute_i"),
    ).with_columns(
        ((pl.col("DAY") - 1).cast(pl.Int64) * SECONDS_PER_DAY
         + pl.col("hour_i").cast(pl.Int64) * 3600
         + pl.col("minute_i").cast(pl.Int64) * 60).alias("ts")
    )
    n_bad = df.filter(pl.col("ts").is_null() | pl.col("SALES_VALUE").is_null()).height
    if n_bad:
        print(f"[prep_cj] dropping {n_bad} rows with unparseable time/amount")
        df = df.filter(pl.col("ts").is_not_null() & pl.col("SALES_VALUE").is_not_null())
    df = df.sort(["household_key", "ts"], maintain_order=True)

    day = df["DAY"].to_numpy().astype(np.int32)
    cut_day = int(np.quantile(day, 1.0 - args.post_frac))
    cut_ts = (cut_day - 1) * SECONDS_PER_DAY
    max_day = int(day.max())
    corpus_cut_date = f"day_{cut_day}_of_{max_day}"  # no calendar anchor in this corpus
    ts = df["ts"].to_numpy().astype(np.int64)
    pre = ts < cut_ts
    print(f"[prep_cj] cut T = {corpus_cut_date} | pre-T rows {int(pre.sum())} "
          f"| post-T rows {int((~pre).sum())}")

    # ---- categorical string fields (raw values) ----
    df = df.with_columns(
        ((pl.col("DAY") - 1) % 7).cast(pl.Utf8).alias("dow"),          # cycle position
        pl.col("hour_i").cast(pl.Utf8).alias("hour_s"),
        ((pl.col("WEEK_NO") - 1) // WEEK_BUCKET).cast(pl.Utf8).alias("week_s"),
        clip_small_int(pl.col("QUANTITY"), QUANTITY_MAX).alias("quantity_s"),
        pl.col("DEPARTMENT").alias("department_s"),
        pl.col("COMMODITY_DESC").alias("commodity_s"),
        pl.col("BRAND").alias("brand_s"),
    )

    # SALES_VALUE quantile buckets - edges from PRE-cut rows only. SALES_VALUE is in
    # dollars, heavily right-skewed with a mass of small values; quantile bucketing
    # needs no distributional assumption.
    amount = df["SALES_VALUE"].to_numpy().astype(np.float32)
    amount_q, amount_edges = quantile_bucket(amount, pre, AMOUNT_BUCKETS)
    # RETAIL_DISC is 0 on most rows and negative otherwise (a markdown); the quantile
    # edges are heavily tied at 0, so most buckets collapse and the nonzero tail gets
    # the resolution. That is the intended behavior for a mass-at-zero column.
    # Nulls are NOT tolerated here: np.quantile propagates a single NaN into every
    # edge, which would silently put all rows in one bucket.
    assert df["RETAIL_DISC"].null_count() == 0, \
        "RETAIL_DISC has nulls; a single null collapses every quantile edge to NaN"
    rdisc = df["RETAIL_DISC"].to_numpy().astype(np.float32)
    rdisc_q, rdisc_edges = quantile_bucket(rdisc, pre, RETAIL_DISC_BUCKETS)

    raw_cols = {
        "dow": "dow", "hour": "hour_s", "week": "week_s", "quantity": "quantity_s",
        "department": "department_s", "commodity": "commodity_s", "brand": "brand_s",
    }
    for col in CJ_FORBIDDEN_COLUMNS:
        assert col not in raw_cols.values() and col not in CJ_FIELDS, \
            f"LEAKAGE: {col} would enter vocab"

    pre_df = df.with_row_index("_i").filter(pl.Series(pre))
    vocabs: dict[str, dict[str, int]] = {}
    token_cols: dict[str, np.ndarray] = {}
    for field, col in raw_cols.items():
        vc = pre_df[col].value_counts(sort=True)
        vals = vc[col].to_list()
        if field == "commodity":
            vals = vals[:COMMODITY_TOP_K]
        mapping = {v: N_SPECIAL + i for i, v in enumerate(vals)}
        vocabs[field] = mapping
        token_cols[field] = (
            df[col].replace_strict(mapping, default=UNK_ID, return_dtype=pl.Int32).to_numpy()
        )
    token_cols["amount_q"] = (N_SPECIAL + amount_q).astype(np.int32)
    token_cols["retail_disc_q"] = (N_SPECIAL + rdisc_q).astype(np.int32)
    bucket_sizes = {"amount_q": AMOUNT_BUCKETS, "retail_disc_q": RETAIL_DISC_BUCKETS}
    vocab_sizes = {f: (N_SPECIAL + bucket_sizes[f]) if f in bucket_sizes
                   else (N_SPECIAL + len(vocabs[f])) for f in CJ_FIELDS}

    tokens = np.stack([token_cols[f] for f in CJ_FIELDS], axis=1)
    assert tokens.max() < 32000, "vocab too large for int16"
    tokens = tokens.astype(np.int16)

    # ---- labels / auxiliary arrays (NOT model inputs) ----
    # fraud.npy: name fixed by common.load_prep's file contract. Here it holds the
    # coupon-discount-present row label (COUPON_DISC < 0), outcome family, label only.
    fraud = (df["COUPON_DISC"].to_numpy() < 0).astype(np.int8)
    # household_key is a small integer id (1..2500); use it directly, mirroring
    # prep.py's raw-int handling of User (sequence index only, never in the vocab).
    # user.npy therefore holds the RAW 1-based key: arrays sized int(user.max()) + 1
    # index correctly (slot 0 unused), but REPORTED household counts must come from
    # n_users / meta['cj']['n_households'], never from max+1.
    user = df["household_key"].to_numpy().astype(np.int32)

    # next-commodity classes from PRE-cut frequency; top-K + other; unseen -> other
    com_vc = pre_df["commodity_s"].value_counts(sort=True)
    com_vals = com_vc["commodity_s"].to_list()[:COMMODITY_TOP_CLASSES]
    com_map = {v: i for i, v in enumerate(com_vals)}
    mcc_class = (
        df["commodity_s"].replace_strict(com_map, default=COMMODITY_TOP_CLASSES,
                                         return_dtype=pl.Int32)
        .to_numpy().astype(np.int16)
    )

    # store key (pooling index for embed.py's merchant path; never in vocab)
    merch_cat = df["STORE_ID"].cast(pl.Utf8).fill_null("NA_STORE").cast(pl.Categorical)
    merch = merch_cat.to_physical().to_numpy().astype(np.int32)
    merch_keys = (
        df.select(pl.col("STORE_ID").cast(pl.Utf8).fill_null("NA_STORE").alias("store_id"))
        .with_columns(pl.Series("merchant_id", merch))
        .unique(subset=["merchant_id"])
        .sort("merchant_id")
        .select("merchant_id", "store_id")
    )

    basket = (df["BASKET_ID"].cast(pl.Utf8).cast(pl.Categorical)
              .to_physical().to_numpy().astype(np.int64))

    # household code -> raw household_key mapping (join key for the offer head; the raw
    # key itself never enters the vocab, same rule as the store key)
    hh_keys = (
        df.select(pl.col("household_key"))
        .with_columns(pl.Series("user", user))
        .unique(subset=["user"])
        .sort("user")
        .select("user", "household_key")
    )

    np.save(out / "tokens.npy", tokens)
    np.save(out / "user.npy", user)
    np.save(out / "ts.npy", ts)
    np.save(out / "fraud.npy", fraud)
    np.save(out / "mcc_class.npy", mcc_class)
    np.save(out / "amount.npy", amount)
    np.save(out / "merchant.npy", merch)
    np.save(out / "basket.npy", basket)   # extra array for the heads (basket controls)
    np.save(out / "day.npy", day)         # extra array for the heads (window bookkeeping)
    merch_keys.write_parquet(out / "merchant_keys.parquet")
    hh_keys.write_parquet(out / "household_keys.parquet")

    sha_txn = "skipped" if args.skip_sha else sha256_file(args.txn)
    sha_prod = "skipped" if args.skip_sha else sha256_file(args.product)

    meta = {
        "fields": CJ_FIELDS,
        "vocab_sizes": [int(vocab_sizes[f]) for f in CJ_FIELDS],
        "amount_edges": [float(e) for e in amount_edges],
        "retail_disc_edges": [float(e) for e in rdisc_edges],
        "cut_ts": cut_ts,
        "cut_day": cut_day,
        "corpus_cut_date": corpus_cut_date,
        "post_frac": args.post_frac,
        "n_rows": int(len(user)),
        "n_users": int(len(np.unique(user))),
        "n_merchants": int(merch_keys.height),
        "mcc_top_k": COMMODITY_TOP_CLASSES,
        "n_mcc_classes": COMMODITY_TOP_CLASSES + 1,
        "sample_stride": args.sample_stride,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "versions": versions_dict(),
        "data_sources": [
            {"name": "dunnhumby The Complete Journey (real retail transactions)",
             "url": CJ_SOURCE_URL, "sha256": CJ_ZIP_SHA256,
             "terms": CJ_TERMS},
        ],
        "leakage": {
            "label_excluded": True,   # coupon columns never tokenized (asserted above)
            "ids_excluded": True,     # household/basket/product/store ids never tokenized
            "vocab_built_pre_cut_only": True,
        },
        "cj": {
            "corpus": "transaction_data.csv joined to product.csv on PRODUCT_ID",
            "source_txn": str(args.txn), "source_txn_sha256": sha_txn,
            "source_product": str(args.product), "source_product_sha256": sha_prod,
            "n_rows_read": int(n_read),
            "n_rows_no_product_match": int(n_no_product),
            "n_households": int(len(np.unique(user))),
            "n_stores": int(merch_keys.height),
            "field_semantics": {
                "dow": "(DAY-1) % 7, a 7-day cycle position (day 1 is not anchored to a weekday)",
                "hour": "TRANS_TIME // 100, clipped 0..23",
                "week": f"(WEEK_NO-1) // {WEEK_BUCKET}, {WEEK_BUCKET}-week buckets",
                "amount_q": f"SALES_VALUE, {AMOUNT_BUCKETS} pre-cut quantile buckets",
                "quantity": f"QUANTITY clipped to 0..{QUANTITY_MAX} (neg / {QUANTITY_MAX + 1}plus buckets)",
                "department": "product.csv DEPARTMENT",
                "commodity": f"product.csv COMMODITY_DESC, top-{COMMODITY_TOP_K} then UNK",
                "brand": "product.csv BRAND (National / Private)",
                "retail_disc_q": f"RETAIL_DISC, {RETAIL_DISC_BUCKETS} pre-cut quantile buckets "
                                 "(mass at zero; buckets collapse on the ties by design)",
            },
            "array_semantics": {
                "fraud.npy": "COUPON_DISC < 0 (coupon discount present on the row); "
                             "outcome family, label only, never an input; the file name "
                             "is fixed by common.load_prep's contract",
                "mcc_class.npy": f"next-commodity class: top-{COMMODITY_TOP_CLASSES} "
                                 "COMMODITY_DESC by pre-cut frequency + 'other'",
                "merchant.npy": "STORE_ID pooling code (embed.py merchant path)",
                "basket.npy": "BASKET_ID code (heads only, never an input)",
                "day.npy": "raw DAY (heads only, never an input)",
                "household_keys.parquet": "user -> household_key (identity: user.npy IS "
                                          "the raw key; kept as the offer head's stable "
                                          "join contract; never an input)",
            },
            "excluded_outcome_like": {
                "COUPON_DISC": "coupon redemption outcome family; cj_offer_head.py's label",
                "COUPON_MATCH_DISC": "same family (retailer match on a coupon)",
            },
            "excluded_identifiers": {
                "household_key": "sequence index only",
                "STORE_ID": "pooling/join key only (merchant.npy)",
                "BASKET_ID": "identifier; saved as basket.npy for head controls",
                "PRODUCT_ID": "identifier; products enter as department/commodity/brand",
            },
            "included_with_rationale": {
                "RETAIL_DISC": "shelf-price markdown set before the purchase, observable "
                               "at transaction time; an input, not an outcome",
            },
            "time_note": "ts is pseudo-epoch seconds (DAY has no calendar anchor); the "
                         "corpus cut is the DAY threshold recorded in cut_day",
        },
    }
    atomic_write_json(out / "meta.json", meta)
    print(f"[prep_cj] done: {len(user)} rows, {meta['n_users']} households, "
          f"{meta['n_merchants']} stores -> {out}")


if __name__ == "__main__":
    main()
