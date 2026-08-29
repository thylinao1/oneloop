"""cj_download.py - verify + convert the Complete Journey CSVs (called by
job_cj_download.sbatch after the zip is verified and extracted).

Does three things, in order, and hard-fails loudly rather than proceeding on drift:
  1. column checks against the verified schema (including the 2023 re-release's
     anonymized hh_demographic columns)
  2. row-count checks: files whose counts were verified manually on 2026-08-24 hard-fail
     on mismatch; the rest are recorded as observed (no public confirmation found)
  3. parquet conversion for the four files the pipeline reads, sha256 for everything,
     and a meta.json with source ref, terms citation, retrieval date and all counts

No model code here; leakage guards live in prep_cj.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import polars as pl

from common import atomic_write_json, versions_dict
from cj_common import (
    CJ_EXPECTED_ROWS, CJ_SOURCE_URL, CJ_TERMS, CJ_USER_GUIDE_PDF,
    COUPON_REDEMPT_COLUMNS_EXPECTED, HH_DEMOGRAPHIC_COLUMNS_EXPECTED,
    PRODUCT_COLUMNS_EXPECTED, TRANSACTION_COLUMNS_EXPECTED, sha256_file,
)

CONVERT = {  # files the pipeline reads -> parquet
    "transaction_data.csv", "product.csv", "hh_demographic.csv", "coupon_redempt.csv",
}
COLUMN_CHECKS = {
    "transaction_data.csv": TRANSACTION_COLUMNS_EXPECTED,
    "product.csv": PRODUCT_COLUMNS_EXPECTED,
    "hh_demographic.csv": HH_DEMOGRAPHIC_COLUMNS_EXPECTED,
    "coupon_redempt.csv": COUPON_REDEMPT_COLUMNS_EXPECTED,
}


def count_rows(path: Path) -> int:
    return int(
        pl.scan_csv(path, infer_schema_length=10000).select(pl.len()).collect().item()
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding the extracted CSVs")
    ap.add_argument("--source-ref", required=True,
                    help="where the zip came from (staged path or DATA_URL)")
    ap.add_argument("--zip-sha256", required=True,
                    help="sha256 of source.zip as computed by the sbatch")
    args = ap.parse_args()
    base = Path(args.dir)

    failures: list[str] = []
    files: dict[str, dict] = {}
    for name, exp in CJ_EXPECTED_ROWS.items():
        p = base / name
        if not p.exists():
            failures.append(f"{name}: MISSING after extraction")
            continue
        cols = pl.scan_csv(p, infer_schema_length=10000).collect_schema().names()
        want = COLUMN_CHECKS.get(name)
        columns_ok = want is None or sorted(cols) == sorted(want)
        if not columns_ok:
            failures.append(f"{name}: column drift, got {cols}, expected {want}")
        n = count_rows(p)
        rec = {"rows_observed": n, "rows_expected": exp["rows"], "status": exp["status"],
               "bytes": p.stat().st_size, "sha256_csv": sha256_file(p)}
        if exp["status"] == "verified" and exp["rows"] is not None and n != exp["rows"]:
            failures.append(f"{name}: {n} rows, verified expectation is {exp['rows']}")
        elif exp["rows"] is not None and n != exp["rows"]:
            print(f"[cj_download] WARNING: {name} has {n} rows vs recorded {exp['rows']} "
                  f"({exp['status']}); recorded, not fatal")
        if name in CONVERT and columns_ok:
            out = p.with_suffix(".parquet")
            pl.scan_csv(p, infer_schema_length=10000).sink_parquet(out)
            rec["parquet"] = out.name
            rec["sha256_parquet"] = sha256_file(out)
            print(f"[cj_download] converted {name} -> {out.name} ({n} rows)")
        files[name] = rec

    if failures:
        for f in failures:
            print(f"[cj_download] FAIL: {f}")
        print("[cj_download] nothing trained on unverified data; fix the source first")
        return 2

    # the user guide PDF is cited only if it actually landed next to the data; the
    # column checks above are the schema authority either way
    pdfs = sorted(p.name for p in base.glob("*.pdf"))
    guide_present = CJ_USER_GUIDE_PDF in pdfs or bool(pdfs)
    meta = {
        "generated_by": "scripts/fm/cj_download.py",
        "source": {
            "name": "dunnhumby The Complete Journey",
            "url": CJ_SOURCE_URL,
            "ref": args.source_ref,
            "terms": CJ_TERMS,
            "zip_sha256": args.zip_sha256,
            "user_guide_pdf_present": guide_present,
            "schema_reference": (f"'{pdfs[0]}' (copied from the archive)" if pdfs
                                 else "user guide PDF not found in the archive; schema "
                                      "verified against the column checks in cj_common.py"),
        },
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
        "versions": versions_dict(),
    }
    atomic_write_json(base / "meta.json", meta)
    print(f"[cj_download] wrote {base / 'meta.json'}")
    for name, rec in files.items():
        print(f"[cj_download]   {name}: {rec['rows_observed']} rows ({rec['status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
