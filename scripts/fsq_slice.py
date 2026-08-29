#!/usr/bin/env python3
"""WS-B3 data prep: Singapore slice of Foursquare OS Places (gated HF dataset).

Streams via DuckDB httpfs with projection+filter pushdown; the 239GB dataset is
NEVER materialized. Output: ~/amex-oneloop/fsq_sg.parquet + fsq_sg.slice_meta.json.

Release pinned to dt=2026-08-11 (latest as of 2026-08-22; verified via HF API).
Schema of that release verified by DESCRIBE on shard places_000000 (2026-08-22):
all COLS below exist with these exact names.

Token: read from ~/amex-oneloop/.hf_token (file may contain either the bare
token or an `HF_TOKEN=hf_...` line). Never printed, never on a command line.
"""
import datetime
import hashlib
import json
import os
import sys
import time

RELEASE = "2026-08-11"
BASE = f"hf://datasets/foursquare/fsq-os-places/release/dt={RELEASE}/places/parquet"
PROBE_FILE = f"{BASE}/places_000000.parquet"
GLOB = f"{BASE}/places_*.parquet"

COLS = [
    "fsq_place_id", "name", "latitude", "longitude", "address", "locality",
    "region", "postcode", "country", "fsq_category_ids", "fsq_category_labels",
    "date_created", "date_closed",
]

OUT_DIR = os.path.expanduser("~/amex-oneloop")
OUT_PARQUET = os.path.join(OUT_DIR, "fsq_sg.parquet")
OUT_META = os.path.join(OUT_DIR, "fsq_sg.slice_meta.json")
TOKEN_PATH = os.path.join(OUT_DIR, ".hf_token")


def read_token(path: str) -> str:
    with open(path) as f:
        raw = f.read().strip()
    tok = raw.split("=", 1)[1].strip() if "=" in raw else raw
    if not tok.startswith("hf_"):
        sys.exit("FATAL: token file does not contain an hf_ token")
    return tok


def main() -> None:
    import duckdb  # installed in the job's /tmp venv

    t0 = time.time()
    token = read_token(TOKEN_PATH)

    scratch = os.environ.get("TMPDIR", "/tmp")
    con = duckdb.connect()
    con.execute(f"SET extension_directory='{scratch}/duckdb_ext'")
    con.execute(f"SET temp_directory='{scratch}/duckdb_tmp'")
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    con.execute(f"SET threads={threads}")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    # Token interpolated into SQL in-process only (CREATE SECRET takes no params).
    con.execute(f"CREATE SECRET hf (TYPE huggingface, TOKEN '{token}')")

    # ---- Fail-fast: verify path + schema on one shard before the big scan ----
    print(f"[fsq_slice] probing {PROBE_FILE}", flush=True)
    desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{PROBE_FILE}')").fetchall()
    have = {row[0] for row in desc}
    missing = [c for c in COLS if c not in have]
    if missing:
        sys.exit(f"FATAL: release dt={RELEASE} is missing expected columns: {missing}")
    print(f"[fsq_slice] probe OK; {len(have)} columns, all {len(COLS)} needed present", flush=True)

    # ---- The slice ----
    col_sql = ", ".join(COLS)
    print(f"[fsq_slice] scanning {GLOB} WHERE country='SG' AND date_closed IS NULL", flush=True)
    con.execute(
        f"""
        COPY (
            SELECT {col_sql}
            FROM read_parquet('{GLOB}')
            WHERE country = 'SG' AND date_closed IS NULL
        ) TO '{OUT_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    elapsed = time.time() - t0
    print(f"[fsq_slice] slice written in {elapsed:.0f}s", flush=True)

    # ---- Sanity checks on the output ----
    n = con.execute(f"SELECT count(*) FROM read_parquet('{OUT_PARQUET}')").fetchone()[0]
    n_cat = con.execute(
        f"SELECT count(*) FROM read_parquet('{OUT_PARQUET}') "
        "WHERE fsq_category_labels IS NOT NULL AND len(fsq_category_labels) > 0"
    ).fetchone()[0]
    countries = con.execute(
        f"SELECT DISTINCT country FROM read_parquet('{OUT_PARQUET}')"
    ).fetchall()
    sample = con.execute(
        f"SELECT name, locality, postcode, fsq_category_labels "
        f"FROM read_parquet('{OUT_PARQUET}') USING SAMPLE 5 ROWS"
    ).fetchall()
    size = os.path.getsize(OUT_PARQUET)
    sha = hashlib.sha256()
    with open(OUT_PARQUET, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)

    print(f"[fsq_slice] rows={n} with_category_labels={n_cat} countries={countries}", flush=True)
    print(f"[fsq_slice] size={size}B sha256={sha.hexdigest()}", flush=True)
    for row in sample:
        print(f"[fsq_slice] sample: {row}", flush=True)

    if n < 50_000:
        sys.exit(f"FATAL: only {n} SG rows; expected roughly 100k-400k; refusing to bless output")
    if countries != [("SG",)]:
        sys.exit(f"FATAL: unexpected countries in output: {countries}")

    meta = {
        "dataset": "foursquare/fsq-os-places (gated HF, Apache-2.0)",
        "release": f"dt={RELEASE}",
        "filter": "country='SG' AND date_closed IS NULL",
        "columns": COLS,
        "rows": n,
        "rows_with_category_labels": n_cat,
        "size_bytes": size,
        "sha256": sha.hexdigest(),
        "duckdb_version": duckdb.__version__,
        "generated_by": "scripts/fsq_slice.py",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "elapsed_seconds": round(elapsed, 1),
        "labels": ["real-public-data"],
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[fsq_slice] DONE; meta at {OUT_META}", flush=True)


if __name__ == "__main__":
    main()
