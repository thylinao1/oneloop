"""make_subset.py: earliest-time-consistent TabFormer subset for the scaling curve.

Takes the N earliest transactions (global timeline order, deterministic
tie-break) and writes them back as a CSV with the ORIGINAL columns, so the
unmodified scripts/fm/prep.py runs on it unchanged: the 0.82-quantile time cut,
pre-cut-only vocab build, and pre-cut amount edges are all recomputed WITHIN the
subset; the time-cutoff logic stays valid at every scale point.

Rows prep.py would drop (unparseable date/amount) are dropped here first, so
the requested N equals the rows prep actually keeps.
"""
from __future__ import annotations

import argparse
import datetime as dt

import polars as pl

SCHEMA = {
    "User": pl.Int64, "Card": pl.Int64, "Year": pl.Int32, "Month": pl.Int32,
    "Day": pl.Int32, "Time": pl.Utf8, "Amount": pl.Utf8, "Use Chip": pl.Utf8,
    "Merchant Name": pl.Utf8, "Merchant City": pl.Utf8, "Merchant State": pl.Utf8,
    "Zip": pl.Utf8, "MCC": pl.Int64, "Errors?": pl.Utf8, "Is Fraud?": pl.Utf8,
}
HELPER_COLS = ["_hour", "_minute", "_amt", "_dtm", "_ts", "_idx"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pl.read_csv(args.csv, schema_overrides=SCHEMA)
    n_raw = df.height
    original_cols = df.columns

    # mirror prep.py's parse + drop so subset row count == prep row count
    df = df.with_columns(
        pl.col("Time").str.slice(0, 2).cast(pl.Int32, strict=False).alias("_hour"),
        pl.col("Time").str.slice(3, 2).cast(pl.Int32, strict=False).alias("_minute"),
        pl.col("Amount").str.replace_all(r"[$,]", "").cast(pl.Float64, strict=False).alias("_amt"),
    ).with_columns(
        pl.datetime("Year", "Month", "Day", "_hour", "_minute").alias("_dtm")
    )
    df = df.filter(pl.col("_dtm").is_not_null() & pl.col("_amt").is_not_null())
    n_parseable = df.height
    df = df.with_columns((pl.col("_dtm").cast(pl.Int64) // 1_000_000).alias("_ts"))

    assert args.rows <= n_parseable, (
        f"requested {args.rows} rows but only {n_parseable} parseable rows exist"
    )
    # deterministic earliest-time subset: sort by (ts, original row order), head N
    df = df.with_row_index("_idx").sort(["_ts", "_idx"]).head(args.rows)
    t_min, t_max = int(df["_ts"].min()), int(df["_ts"].max())
    sub = df.drop(HELPER_COLS).select(original_cols)
    assert sub.height == args.rows, f"subset height {sub.height} != requested {args.rows}"
    sub.write_csv(args.out)

    fmt = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(
        f"[subset] {args.rows} earliest-time rows of {n_parseable} parseable "
        f"({n_raw} raw) | timeline {fmt(t_min)} .. {fmt(t_max)} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
