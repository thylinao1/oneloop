#!/usr/bin/env python3
"""pair_retention.py: the merchant-native task the merchant axis was never asked.

Pre-registered in the project record BEFORE the first run.

Unit of analysis: an (account, merchant) pair that transacted at least once
strictly BEFORE the corpus cut. Label: 1 if the same pair transacts again at or
after the cut. Split: entity-disjoint by MERCHANT (seeded hash partition).

Arms (identical model settings, only the feature block moves):
  baseline   pair frequency + recency + account/merchant pre-cut activity
  with_v2    baseline + 512-d MERCHANT-AXIS pooled embedding   (primary)
  with_v1    baseline + 512-d CARDHOLDER-AXIS pooled embedding (secondary)

Stages:
  --stage pairs  stream the TabFormer CSV out of the tgz, emit the pair table
  --stage run    features + LightGBM + merchant-clustered paired bootstrap
  --check        recompute every numeric leaf from the cached pair table and
                 compare against the committed results file at 1e-6
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
TGZ = ROOT / "data" / "transactions.tgz"
TGZ_SHA = "e9f589a0958f40d60f81b1a2e8428db86e00c05755caf44fb055827976c0efa2"
EMB_V2 = ROOT / "data" / "merchant_embeddings_v2.parquet"
EMB_V1 = ROOT / "data" / "merchant_embeddings.parquet"
PAIRS = ROOT / "results" / "cache" / "pair_retention_pairs.parquet"
OUT = ROOT / "results" / "pair_retention.json"

# committed prep cut, read from the backbone meta and never recomputed here
CUT_TS = 1503653820
CUT_DATE = "2017-08-25T09:37:00Z"
CORPUS_ROWS = 24386900

SEED = 7
TEST_FRAC = 0.30
MAX_TRAIN = 2_000_000
MAX_TEST = 800_000
BOOTSTRAP = 1000
DAY = 86400.0

BASE_FEATURES = [
    "pair_n_pre", "pair_days_since_last", "pair_days_since_first", "pair_span_days",
    "pair_mean_gap_days", "pair_amt_sum", "pair_amt_mean",
    "acct_n_pre", "acct_n_merch_pre",
    "merch_n_pre", "merch_n_acct_pre", "merch_days_since_last",
    "pair_share_of_acct",
]


# --------------------------------------------------------------------------- #
# stage: pairs
# --------------------------------------------------------------------------- #
def merchant_key_frame() -> pl.DataFrame:
    """merchant_id <- (Merchant Name, city) from the committed embedding table."""
    return pl.read_parquet(EMB_V2, columns=["merchant_id", "merchant_name", "merchant_city"])


def stage_pairs() -> None:
    import pyarrow.csv as pacsv

    assert TGZ.exists(), f"missing {TGZ}"
    print("[pairs] verifying corpus sha256 ...", flush=True)
    h = hashlib.sha256()
    with open(TGZ, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    got = h.hexdigest()
    assert got == TGZ_SHA, f"corpus sha256 mismatch: {got}"
    print(f"[pairs] corpus sha256 OK {got}", flush=True)

    keys = merchant_key_frame()
    print(f"[pairs] merchant key table: {keys.height} merchants", flush=True)

    tf = tarfile.open(TGZ, "r:gz")
    member = tf.getmember("card_transaction.v1.csv")
    stream = tf.extractfile(member)
    reader = pacsv.open_csv(
        stream,
        read_options=pacsv.ReadOptions(block_size=1 << 25),
        convert_options=pacsv.ConvertOptions(
            column_types={"User": "int32", "Year": "int32", "Month": "int32",
                          "Day": "int32", "Time": "string", "Amount": "string",
                          "Merchant Name": "string", "Merchant City": "string"},
            include_columns=["User", "Year", "Month", "Day", "Time", "Amount",
                             "Merchant Name", "Merchant City"],
        ),
    )

    pre_parts: list[pl.DataFrame] = []
    post_parts: list[pl.DataFrame] = []
    n_rows = 0
    n_unmatched = 0
    n_unmatched_pre = 0
    t0 = time.time()
    for batch in reader:
        df = pl.from_arrow(batch)
        n_rows += df.height
        df = df.with_columns(
            pl.col("Time").str.slice(0, 2).cast(pl.Int32, strict=False).alias("hh"),
            pl.col("Time").str.slice(3, 2).cast(pl.Int32, strict=False).alias("mm"),
            pl.col("Amount").str.replace_all(r"[$,]", "").cast(pl.Float64, strict=False).alias("amt"),
            pl.col("Merchant City").fill_null("NA").alias("merchant_city"),
        ).with_columns(
            (pl.datetime("Year", "Month", "Day", "hh", "mm").cast(pl.Int64) // 1_000_000).alias("ts")
        ).drop_nulls(["ts", "amt"])
        df = df.rename({"Merchant Name": "merchant_name"}).join(
            keys, on=["merchant_name", "merchant_city"], how="left"
        )
        miss = df.filter(pl.col("merchant_id").is_null())
        n_unmatched += miss.height
        n_unmatched_pre += int((miss["ts"] < CUT_TS).sum())
        df = df.drop_nulls("merchant_id").select(
            pl.col("User").alias("acct"), "merchant_id", "ts", "amt"
        )
        pre = df.filter(pl.col("ts") < CUT_TS)
        post = df.filter(pl.col("ts") >= CUT_TS)
        if pre.height:
            pre_parts.append(pre.group_by(["acct", "merchant_id"]).agg(
                pl.len().alias("n"), pl.col("ts").min().alias("t_first"),
                pl.col("ts").max().alias("t_last"), pl.col("amt").sum().alias("amt_sum")))
        if post.height:
            post_parts.append(post.group_by(["acct", "merchant_id"]).agg(
                pl.len().alias("n_post")))
        if len(pre_parts) >= 24:
            pre_parts = [_fold_pre(pre_parts)]
        if len(post_parts) >= 24:
            post_parts = [_fold_post(post_parts)]
        if n_rows % 4_000_000 < (1 << 22):
            print(f"[pairs] {n_rows:,} rows  {time.time()-t0:.0f}s", flush=True)
    stream.close()
    tf.close()

    print(f"[pairs] rows read {n_rows:,} (corpus {CORPUS_ROWS:,}), "
          f"unmatched-merchant rows {n_unmatched:,} of which pre-cut {n_unmatched_pre:,}", flush=True)
    # ASSERT: an unmatched merchant is one with no pre-cut transaction, so it can
    # never sit in a pre-cut pair. A pre-cut unmatched row would mean the join is
    # losing real pairs, which is a hard failure, not a rounding note.
    assert n_unmatched_pre == 0, f"unmatched merchant on {n_unmatched_pre} PRE-cut rows"
    pre_agg = _fold_pre(pre_parts)
    post_agg = _fold_post(post_parts)
    print(f"[pairs] distinct pre-cut pairs {pre_agg.height:,}; "
          f"distinct post-cut pairs {post_agg.height:,}", flush=True)

    pairs = pre_agg.join(post_agg, on=["acct", "merchant_id"], how="left").with_columns(
        pl.col("n_post").fill_null(0)
    ).with_columns((pl.col("n_post") > 0).cast(pl.Int8).alias("y"))
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    pairs.write_parquet(PAIRS)
    meta = {
        "n_rows_read": n_rows, "n_unmatched_merchant_rows": n_unmatched,
        "n_unmatched_merchant_rows_pre_cut": n_unmatched_pre,
        "n_pairs": pairs.height, "pos_rate": float(pairs["y"].mean()),
        "cut_ts": CUT_TS, "cut_date": CUT_DATE,
    }
    (PAIRS.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=1))
    print(f"[pairs] wrote {PAIRS} :: {meta}", flush=True)


def _fold_pre(parts):
    return pl.concat(parts).group_by(["acct", "merchant_id"]).agg(
        pl.col("n").sum().alias("n"), pl.col("t_first").min().alias("t_first"),
        pl.col("t_last").max().alias("t_last"), pl.col("amt_sum").sum().alias("amt_sum"))


def _fold_post(parts):
    return pl.concat(parts).group_by(["acct", "merchant_id"]).agg(
        pl.col("n_post").sum().alias("n_post"))


# --------------------------------------------------------------------------- #
# stage: run
# --------------------------------------------------------------------------- #
def build_features(pairs: pl.DataFrame) -> pl.DataFrame:
    acct = pairs.group_by("acct").agg(
        pl.col("n").sum().alias("acct_n_pre"), pl.len().alias("acct_n_merch_pre"))
    merch = pairs.group_by("merchant_id").agg(
        pl.col("n").sum().alias("merch_n_pre"), pl.len().alias("merch_n_acct_pre"),
        pl.col("t_last").max().alias("merch_t_last"))
    df = pairs.join(acct, on="acct").join(merch, on="merchant_id")
    return df.with_columns(
        pl.col("n").cast(pl.Float32).alias("pair_n_pre"),
        ((CUT_TS - pl.col("t_last")) / DAY).cast(pl.Float32).alias("pair_days_since_last"),
        ((CUT_TS - pl.col("t_first")) / DAY).cast(pl.Float32).alias("pair_days_since_first"),
        ((pl.col("t_last") - pl.col("t_first")) / DAY).cast(pl.Float32).alias("pair_span_days"),
        pl.when(pl.col("n") > 1)
          .then((pl.col("t_last") - pl.col("t_first")) / DAY / (pl.col("n") - 1))
          .otherwise(None).cast(pl.Float32).alias("pair_mean_gap_days"),
        pl.col("amt_sum").cast(pl.Float32).alias("pair_amt_sum"),
        (pl.col("amt_sum") / pl.col("n")).cast(pl.Float32).alias("pair_amt_mean"),
        pl.col("acct_n_pre").cast(pl.Float32),
        pl.col("acct_n_merch_pre").cast(pl.Float32),
        pl.col("merch_n_pre").cast(pl.Float32),
        pl.col("merch_n_acct_pre").cast(pl.Float32),
        ((CUT_TS - pl.col("merch_t_last")) / DAY).cast(pl.Float32).alias("merch_days_since_last"),
        (pl.col("n") / pl.col("acct_n_pre")).cast(pl.Float32).alias("pair_share_of_acct"),
    )


def merchant_hash_split(mids: np.ndarray, seed: int, test_frac: float) -> np.ndarray:
    """Deterministic per-merchant assignment; no merchant can land on both sides."""
    salt = f"pair-retention-{seed}".encode()
    out = np.empty(len(mids), dtype=bool)
    for i, m in enumerate(mids):
        h = hashlib.sha256(salt + int(m).to_bytes(8, "little", signed=True)).digest()
        out[i] = (int.from_bytes(h[:8], "little") / 2 ** 64) < test_frac
    return out


def load_emb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cols = ["merchant_id"] + [f"emb_{i}" for i in range(512)]
    d = pl.read_parquet(path, columns=cols)
    mid = d["merchant_id"].to_numpy()
    emb = d.drop("merchant_id").to_numpy().astype(np.float32)
    return mid, emb


def emb_matrix(mid: np.ndarray, emb: np.ndarray, want: np.ndarray) -> tuple[np.ndarray, int]:
    order = np.argsort(mid)
    pos = np.searchsorted(mid[order], want)
    pos = np.clip(pos, 0, len(mid) - 1)
    src = order[pos]
    hit = mid[src] == want
    out = np.zeros((len(want), emb.shape[1] + 1), dtype=np.float32)
    out[hit, :-1] = emb[src[hit]]
    out[hit, -1] = 1.0
    return out, int((~hit).sum())


def paired_cluster_bootstrap(cluster_of_row, delta_fns, B, seed):
    uniq, inv = np.unique(cluster_of_row, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    bnd = np.append(np.searchsorted(sorted_inv, np.arange(len(uniq))), len(sorted_inv))
    rows_by_c = [order[bnd[i]:bnd[i + 1]] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.full((B, len(delta_fns)), np.nan)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_c[p] for p in pick])
        for k, fn in enumerate(delta_fns):
            try:
                out[b, k] = fn(idx)
            except ValueError:
                pass
    return [[float(np.nanpercentile(out[:, k], 2.5)), float(np.nanpercentile(out[:, k], 97.5))]
            for k in range(len(delta_fns))]


def versions_dict() -> dict:
    import lightgbm, sklearn, pyarrow
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "polars": pl.__version__, "pyarrow": pyarrow.__version__,
            "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__}


def stage_run(check: bool = False) -> None:
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    assert PAIRS.exists(), f"missing {PAIRS}; run --stage pairs first"
    pairs = pl.read_parquet(PAIRS)
    pmeta = json.loads(PAIRS.with_suffix(".meta.json").read_text())
    assert pmeta["cut_ts"] == CUT_TS, "cut drifted from the committed prep cut"

    df = build_features(pairs)
    mids = df["merchant_id"].to_numpy()
    y = df["y"].to_numpy().astype(np.int8)
    X_base = df.select(BASE_FEATURES).to_numpy().astype(np.float32)

    uniq_m = np.unique(mids)
    is_test_m = merchant_hash_split(uniq_m, SEED, TEST_FRAC)
    test_m = set(uniq_m[is_test_m].tolist())
    te_mask = np.fromiter((int(m) in test_m for m in mids), dtype=bool, count=len(mids))
    tr_mask = ~te_mask

    # ASSERT: zero merchant overlap (a nonzero overlap is a hard failure)
    n_overlap = len(set(mids[tr_mask].tolist()) & set(mids[te_mask].tolist()))
    assert n_overlap == 0, f"merchant overlap between train and test: {n_overlap}"

    rng = np.random.default_rng(SEED)
    tr_idx = np.flatnonzero(tr_mask)
    te_idx = np.flatnonzero(te_mask)
    n_tr_full, n_te_full = len(tr_idx), len(te_idx)
    if len(tr_idx) > MAX_TRAIN:
        tr_idx = np.sort(rng.choice(tr_idx, MAX_TRAIN, replace=False))
    if len(te_idx) > MAX_TEST:
        te_idx = np.sort(rng.choice(te_idx, MAX_TEST, replace=False))

    # validation split for early stopping, merchant-disjoint inside the train side
    tr_m = np.unique(mids[tr_idx])
    val_m = set(rng.permutation(tr_m)[: max(1, int(len(tr_m) * 0.15))].tolist())
    val_mask = np.fromiter((int(m) in val_m for m in mids[tr_idx]), dtype=bool, count=len(tr_idx))

    want_tr, want_te = mids[tr_idx], mids[te_idx]
    Xb_tr, Xb_te = X_base[tr_idx], X_base[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    params = dict(objective="binary", metric="average_precision", learning_rate=0.05,
                  num_leaves=63, n_estimators=600, seed=SEED, n_jobs=4, verbose=-1)

    # one arm at a time: a 513-column float32 matrix for every arm at once does not
    # fit the 8GB laptop this stage runs on, so each arm is built, trained, scored
    # and released before the next one is materialized.
    scores, iters, nfeat, unmatched = {}, {}, {}, {}
    for arm in ("baseline", "with_v2", "with_v1"):
        if arm == "baseline":
            Xtr, Xte = Xb_tr, Xb_te
        else:
            mid, emb = load_emb(EMB_V2 if arm == "with_v2" else EMB_V1)
            Etr, u1 = emb_matrix(mid, emb, want_tr)
            Ete, u2 = emb_matrix(mid, emb, want_te)
            unmatched[arm] = int(u1 + u2)
            del mid, emb
            Xtr = np.hstack([Xb_tr, Etr]); del Etr
            Xte = np.hstack([Xb_te, Ete]); del Ete
        clf = lgb.LGBMClassifier(**params)
        clf.fit(Xtr[~val_mask], y_tr[~val_mask],
                eval_set=[(Xtr[val_mask], y_tr[val_mask])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        scores[arm] = clf.predict_proba(Xte)[:, 1]
        iters[arm] = int(clf.best_iteration_ or params["n_estimators"])
        nfeat[arm] = int(Xtr.shape[1])
        print(f"[run] {arm}: {iters[arm]} iters, {nfeat[arm]} features", flush=True)
        del clf, Xtr, Xte

    def metrics(s):
        return {"auc": float(roc_auc_score(y_te, s)), "prauc": float(average_precision_score(y_te, s))}

    res_arms = {a: {**metrics(s), "n_features": nfeat[a], "lgbm_iterations": iters[a]}
                for a, s in scores.items()}
    sb = scores["baseline"]
    deltas = {}
    for arm in ("with_v2", "with_v1"):
        se = scores[arm]
        fns = [
            (lambda i, se=se: roc_auc_score(y_te[i], se[i]) - roc_auc_score(y_te[i], sb[i])),
            (lambda i, se=se: average_precision_score(y_te[i], se[i]) - average_precision_score(y_te[i], sb[i])),
        ]
        ci = paired_cluster_bootstrap(want_te, fns, BOOTSTRAP, SEED)
        deltas[arm] = {
            "delta_auc": res_arms[arm]["auc"] - res_arms["baseline"]["auc"],
            "delta_auc_ci": ci[0],
            "delta_prauc": res_arms[arm]["prauc"] - res_arms["baseline"]["prauc"],
            "delta_prauc_ci": ci[1],
            "unmatched_pairs": unmatched[arm],
        }
        print(f"[run] {arm} delta_auc {deltas[arm]['delta_auc']:+.6f} {ci[0]} | "
              f"delta_prauc {deltas[arm]['delta_prauc']:+.6f} {ci[1]}", flush=True)

    out = {
        "seed": SEED,
        "versions": versions_dict(),
        "generated_by": "scripts/pair_retention.py --check-able",
        "data_sources": [
            {"name": "IBM TabFormer card_transaction.v1.csv (synthetic)",
             "url": "https://github.com/IBM/TabFormer (data/credit_card/transactions.tgz)",
             "sha256": TGZ_SHA},
            {"name": "merchant-axis pooled merchant embeddings (merchant_embeddings_v2.parquet)",
             "url": "cluster artifact of the merchant-axis pretraining run",
             "sha256": "670372ae31440b9220361a40f0cc51a51c6729f9807ceb88ce37346aa14711a4"},
        ],
        "labels": ["synthetic"],
        "what_this_is": (
            "The merchant-native downstream task. The merchant axis was previously evaluated only on "
            "next merchant category and fraud, both of which are near-degenerate when the entity is the "
            "merchant. Pair retention is not: the unit is an (account, merchant) pair seen before the "
            "cut, and the label is whether that pair transacts again after it. Pre-registered in "
            "the project record before the first run."),
        "task": {
            "unit": "(account, merchant) pair with at least one transaction strictly before the cut",
            "label": "1 if the same pair transacts again at or after the cut",
            "cut_ts": CUT_TS, "cut_date": CUT_DATE,
            "n_pairs_total": int(pmeta["n_pairs"]),
            "pos_rate_total": float(pmeta["pos_rate"]),
            "n_rows_read": int(pmeta["n_rows_read"]),
            "n_unmatched_merchant_rows": int(pmeta["n_unmatched_merchant_rows"]),
            "n_unmatched_merchant_rows_pre_cut": int(pmeta["n_unmatched_merchant_rows_pre_cut"]),
        },
        "split": {
            "kind": "entity-disjoint by merchant (seeded sha256 hash partition)",
            "test_frac": TEST_FRAC,
            "n_merchants_train": int(len(uniq_m) - int(is_test_m.sum())),
            "n_merchants_test": int(is_test_m.sum()),
            "n_shared_merchants_train_test": int(n_overlap),
            "n_train_pairs_available": int(n_tr_full),
            "n_test_pairs_available": int(n_te_full),
            "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "n_test_merchants_scored": int(len(np.unique(want_te))),
            "test_pos_rate": float(y_te.mean()),
            "caveat": ("merchant-disjoint only: cardholders cross merchant boundaries, the same "
                       "disclosed caveat the merchant-axis pretraining run carries"),
        },
        "held_fixed": {
            "model": "LightGBM binary, identical settings across all three arms",
            "params": {k: v for k, v in params.items() if k != "n_jobs"},
            "early_stopping": "50 rounds on a merchant-disjoint 15% validation slice of the train side",
            "baseline_features": BASE_FEATURES,
            "embedding_dim": 512,
            "embedding_pooling": "pre-cut transactions only",
        },
        "bootstrap": {"method": "merchant-clustered paired bootstrap", "B": BOOTSTRAP,
                      "ci": "percentile 95%"},
        "arms": res_arms,
        "deltas": deltas,
        "primary": "with_v2",
        "check": "scripts/pair_retention.py --check (recomputes stage run from the cached pair table)",
    }

    if check:
        prev = json.loads(OUT.read_text())
        bad = _compare(prev, out, "")
        if bad:
            print("CHECK FAILED", flush=True)
            for b in bad[:40]:
                print("  ", b, flush=True)
            sys.exit(1)
        print("CHECK OK", flush=True)
        return

    OUT.write_text(json.dumps(out, indent=1))
    print(f"[run] wrote {OUT}", flush=True)


def _compare(a, b, path, tol=1e-6, bad=None):
    if bad is None:
        bad = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in ("versions", "generated_by"):
                continue
            if k not in a or k not in b:
                bad.append(f"{path}/{k}: present in only one file")
            else:
                _compare(a[k], b[k], f"{path}/{k}", tol, bad)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            bad.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, z) in enumerate(zip(a, b)):
                _compare(x, z, f"{path}/{i}", tol, bad)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(float(a) - float(b)) > tol:
            bad.append(f"{path}: {a} vs {b}")
    elif a != b:
        bad.append(f"{path}: {a!r} vs {b!r}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pairs", "run"], default="run")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.stage == "pairs":
        stage_pairs()
    else:
        stage_run(check=args.check)


if __name__ == "__main__":
    main()
