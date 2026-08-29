"""cj_common.py - shared constants + helpers for the Complete Journey replication (W-ELO
workstream, corpus swapped to dunnhumby "The Complete Journey" after the Elo license
check failed; see ELO-DROPPED.md).

Real retail transactions from ~2,500 households over 711 days, released by dunnhumby for
research, personal or non-commercial use. Mirrors common.py so the UNMODIFIED
pretrain.py / embed.py run on this corpus through cj_run.py.

LEAKAGE POLICY (same four hard-fails as common.py, restated for Complete Journey):
  (a) COUPON_DISC and COUPON_MATCH_DISC are outcome-adjacent (the coupon-redemption
      outcome family that cj_offer_head.py predicts) and NEVER enter model inputs/vocab.
      RETAIL_DISC IS an input: it is a shelf-price markdown observable on the receipt at
      transaction time, set by the retailer before the purchase, not an outcome of it.
  (b) pretraining corpus hard-truncated at cut_ts (recorded in every output).
  (c) as-of embeddings use ONLY transactions strictly before the scored one.
  (d) household_key NEVER in the vocab (sequence index only); STORE_ID NEVER in the
      vocab (pooling/join key only, mirroring how prep.py keeps Merchant Name out of the
      vocab and represents merchants by pooled encodings); BASKET_ID and PRODUCT_ID are
      identifiers and never tokenized (products are represented by DEPARTMENT,
      COMMODITY_DESC and BRAND).

WHY patch_common EXISTS: pretrain.py and embed.py never use field NAMES except through
common.load_prep's equality assert against common.FIELDS. Rather than fork those two
scripts (code drift, and a weaker "same code path" claim), cj_run.py installs CJ_FIELDS
into the already-imported common module and then runs the UNMODIFIED pretrain.py /
embed.py. The mutation is process-local, explicit, and happens before any pipeline code
runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# Model-input fields (tokenized). NOTE: no household_key, no BASKET_ID, no PRODUCT_ID,
# no STORE_ID, no COUPON_DISC, no COUPON_MATCH_DISC. Order matters: vocab_sizes and
# token columns follow this order everywhere.
CJ_FIELDS = [
    "dow", "hour", "week", "amount_q", "quantity",
    "department", "commodity", "brand", "retail_disc_q",
]
CJ_FORBIDDEN_COLUMNS = {
    "household_key", "BASKET_ID", "PRODUCT_ID", "STORE_ID",
    "COUPON_DISC", "COUPON_MATCH_DISC",
}

COMMODITY_TOP_K = 300   # top-K then UNK, same spirit as prep.py's CITY_TOP_K; the corpus
                        # has ~308 distinct COMMODITY_DESC values (observed at prep)
AMOUNT_BUCKETS = 100    # same as prep.py (SALES_VALUE quantile buckets, pre-cut edges)
RETAIL_DISC_BUCKETS = 20  # RETAIL_DISC is 0 on most rows and negative otherwise, so a
                          # mass-at-zero column; 20 quantile buckets avoid 80 empty ones
COMMODITY_TOP_CLASSES = 30  # next-commodity task: top-30 classes + 1 'other', mirroring
                            # prep.py's MCC_TOP_K for the MCC task
QUANTITY_MAX = 12       # clip; QUANTITY has extreme outliers (fuel rows record
                        # hundredths of gallons)
WEEK_BUCKET = 4         # WEEK_NO (1..102) bucketed into 4-week groups (~26 tokens)

# VERIFIED constants (the official dunnhumby CDN asset was downloaded and verified manually
# on 2026-08-24; zip staged to the cluster at ~/amex-oneloop/cj/source.zip).
CJ_ZIP_SHA256 = "5e0a3d72fe8562fe0ab995f70fb58b74359e8ec4bbccd1521e2b137da0558f9a"
CJ_SOURCE_URL = "https://www.dunnhumby.com/source-files/"
CJ_TERMS = ("dunnhumby terms: data may be used solely for research, personal or "
            "non-commercial purposes")
# Archive layout inside source.zip (directory names contain spaces; quote in shell):
CJ_ZIP_CSV_DIR = "dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV"
CJ_USER_GUIDE_PDF = "dunnhumby - The Complete Journey User Guide.pdf"

# Row counts: "verified" = counted manually on the downloaded files 2026-08-24
# (hard-fail on mismatch); "observed at download" = recorded by the download job, no
# public confirmation found (warn on drift, never fail).
CJ_EXPECTED_ROWS = {
    "transaction_data.csv": {"rows": 2_595_732, "status": "verified"},
    "product.csv": {"rows": 92_353, "status": "verified"},
    "hh_demographic.csv": {"rows": 801, "status": "verified"},
    "coupon_redempt.csv": {"rows": 2_318, "status": "verified"},
    "campaign_table.csv": {"rows": None, "status": "observed at download"},
    "campaign_desc.csv": {"rows": None, "status": "observed at download"},
    "coupon.csv": {"rows": None, "status": "observed at download"},
    "causal_data.csv": {"rows": None, "status": "observed at download"},
}

TRANSACTION_COLUMNS_EXPECTED = [
    "household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE",
    "STORE_ID", "RETAIL_DISC", "TRANS_TIME", "WEEK_NO", "COUPON_DISC",
    "COUPON_MATCH_DISC",
]
PRODUCT_COLUMNS_EXPECTED = [
    "PRODUCT_ID", "MANUFACTURER", "DEPARTMENT", "BRAND", "COMMODITY_DESC",
    "SUB_COMMODITY_DESC", "CURR_SIZE_OF_PRODUCT",
]
# 2023 re-release anonymized the demographic columns; there are no AGE_DESC or
# INCOME_DESC columns in this asset.
HH_DEMOGRAPHIC_COLUMNS_EXPECTED = [
    "classification_1", "classification_2", "classification_3", "HOMEOWNER_DESC",
    "classification_5", "classification_4", "KID_CATEGORY_DESC", "household_key",
]
COUPON_REDEMPT_COLUMNS_EXPECTED = ["household_key", "DAY", "COUPON_UPC", "CAMPAIGN"]

SECONDS_PER_DAY = 86_400


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def patch_common() -> None:
    """Install the Complete Journey field list into the already-imported common module
    so the UNMODIFIED pretrain.py / embed.py run on a CJ prep. Process-local."""
    import common

    common.FIELDS = list(CJ_FIELDS)
    common.FORBIDDEN_COLUMNS = set(common.FORBIDDEN_COLUMNS) | set(CJ_FORBIDDEN_COLUMNS)


def load_prep_cj(prep_dir: str | Path) -> dict:
    """CJ counterpart of common.load_prep: same array files, CJ field assert.

    Array-name contract note: the loader (and pretrain.py through it) expects files named
    fraud.npy / mcc_class.npy / merchant.npy. On this corpus those hold, respectively,
    the coupon-discount-present row label (outcome family, never an input), the
    next-commodity class label, and the STORE_ID pooling code. meta['cj'] records the
    semantics so no number is ever read under the wrong name.
    """
    p = Path(prep_dir)
    meta = json.loads((p / "meta.json").read_text())
    assert set(meta["fields"]) == set(CJ_FIELDS), "field list drift between prep_cj and code"
    for col in CJ_FORBIDDEN_COLUMNS:
        assert col not in meta["fields"], f"LEAKAGE: {col} in model fields"
    assert meta.get("cj", {}).get("corpus"), "meta.json missing the cj provenance block"
    d = {"meta": meta}
    for name in ("tokens", "user", "ts", "fraud", "mcc_class", "amount", "merchant"):
        d[name] = np.load(p / f"{name}.npy")
    return d
