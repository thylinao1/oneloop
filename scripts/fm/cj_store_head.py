"""cj_store_head.py - the merchant-level demonstration on the Complete Journey corpus.

The shipped limitation L-B: the offers and corridor heads never consumed the backbone.
This head shows a merchant-level (here: store-level) consumer of backbone merchant
embeddings against a REAL outcome, joined through the shared store key that prep_cj.py
kept out of the vocab.

Question (pre-registered in CJ-REPLICATION-PREREG.md): do pre-cut pooled store
embeddings from the backbone add anything over a counts-only control when predicting
each store's post-cut sales growth tercile? A null ships as a null.

Outcome (the pre-registered choice between the two candidates): post-cut sales GROWTH
tercile. growth(store) = (post-cut SALES_VALUE per post day) / (pre-cut SALES_VALUE per
pre day); terciles over eligible stores; three classes low/mid/high. Growth rather than
raw post-cut volume, because raw volume is mostly pre-cut size restated and the counts
control would win by construction without answering the embedding question.

Leakage hard-fails implemented HERE:
  - store embeddings come from embed.py's merchant path, pooled over PRE-cut
    transactions only (asserted via the parquet's n_txns_pre_cut and the prep cut)
  - every control feature is computed from PRE-cut rows only
  - the outcome uses POST-cut rows only; no feature touches the post-cut window
  - store-disjoint split: a store is scored only if it never trained the head

Scale honesty: ~550 stores total, fewer after the eligibility floor. Intervals will be
wide; that is the corpus, not a bug.

Usage: cj_store_head.py --prep PREP --emb store_embeddings_s7.parquet \
         --out results/cj_store_head.json --backbone-seed 7
Check:  add --check to recompute and compare every numeric leaf at 1e-6 (exit 0/5).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm  # noqa: F401  MUST load before torch (seed_everything imports torch;
                 # macOS libomp coexists only if LightGBM's loads first)
import numpy as np
import polars as pl

from common import atomic_write_json, seed_everything, versions_dict
from cj_common import load_prep_cj, sha256_file
from transfer_eval import paired_cluster_bootstrap

MIN_PRE_ROWS = 50       # eligibility floor: stores with fewer pre-cut rows have growth
                        # ratios dominated by noise; recorded, pre-registered
N_CLASSES = 3
CONTROL_FEATURES = ["log1p_pre_txn_count", "log1p_pre_distinct_households",
                    "pre_mean_basket_value"]


def build(args) -> dict:
    from sklearn.metrics import f1_score
    import lightgbm as lgb

    seed_everything(args.seed)
    d = load_prep_cj(args.prep)
    meta = d["meta"]
    pre = d["ts"] < meta["cut_ts"]
    post = ~pre
    merch = d["merchant"]
    basket = np.load(Path(args.prep) / "basket.npy")

    cut_day = int(meta["cut_day"])
    max_day = int(np.load(Path(args.prep) / "day.npy").max())
    n_pre_days = cut_day - 1
    n_post_days = max_day - cut_day + 1
    assert n_pre_days > 0 and n_post_days > 0, "degenerate pre/post windows"

    # ---- per-store PRE-cut controls (counts only, the honest competitor) ----
    df = pl.DataFrame({
        "store": merch, "user": d["user"], "basket": basket,
        "amount": d["amount"].astype(np.float64), "pre": pre,
    })
    pre_g = (
        df.filter(pl.col("pre"))
        .group_by("store")
        .agg(
            pl.len().alias("pre_txn_count"),
            pl.col("user").n_unique().alias("pre_distinct_households"),
            pl.col("basket").n_unique().alias("pre_distinct_baskets"),
            pl.col("amount").sum().alias("pre_sales"),
        )
        .with_columns(
            (pl.col("pre_sales") / pl.col("pre_distinct_baskets")).alias("pre_mean_basket_value")
        )
    )
    post_g = (
        df.filter(~pl.col("pre"))
        .group_by("store")
        .agg(pl.col("amount").sum().alias("post_sales"))
    )
    stores = pre_g.join(post_g, on="store", how="left").with_columns(
        pl.col("post_sales").fill_null(0.0)
    ).with_columns(
        ((pl.col("post_sales") / n_post_days) / (pl.col("pre_sales") / n_pre_days))
        .alias("growth")
    )
    n_stores_total = int(len(np.unique(merch)))
    # the sentinel code for null STORE_ID (if any rows had one) is not a store and is
    # excluded from the universe before eligibility
    keys = pl.read_parquet(Path(args.prep) / "merchant_keys.parquet")
    na_codes = keys.filter(pl.col("store_id") == "NA_STORE")["merchant_id"].to_list()
    is_na = pl.col("store").is_in(na_codes)
    below_floor = (pl.col("pre_txn_count") < args.min_pre_rows) | (pl.col("pre_sales") <= 0)
    eligible = stores.filter(~is_na & ~below_floor).sort("store")
    # exclusion accounting, decomposed (a single "excluded" number would hide that it
    # mixes three different reasons); the sum is asserted, not assumed
    exclusions = {
        "n_stores_total": n_stores_total,
        "n_post_cut_only": int(n_stores_total - stores.height),  # no pre-cut rows at all
        "n_na_store_key": int(stores.filter(is_na).height),
        "n_below_floor": int(stores.filter(~is_na & below_floor).height),
        "n_eligible": int(eligible.height),
    }
    assert (exclusions["n_post_cut_only"] + exclusions["n_na_store_key"]
            + exclusions["n_below_floor"] + exclusions["n_eligible"]) == n_stores_total, \
        "store exclusion accounting does not sum to the universe"

    growth = eligible["growth"].to_numpy()
    edges = np.quantile(growth, [1 / 3, 2 / 3])  # tercile cuts over eligible stores

    # ---- backbone store embeddings (pre-cut pooled, embed.py merchant path) ----
    emb_df = pl.read_parquet(args.emb)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    assert emb_cols, "no emb_* columns in the embeddings parquet"
    emb_df = emb_df.filter(pl.col("store_id") != "NA_STORE")
    joined = eligible.join(
        emb_df.select(pl.col("merchant_id").alias("store"), "n_txns_pre_cut", *emb_cols),
        on="store", how="inner",
    ).sort("store")
    assert joined.height == eligible.height, (
        f"{eligible.height - joined.height} eligible stores missing an embedding; "
        "eligibility requires pre-cut rows, so pooling should cover them all")
    # free contamination detector: embed.py's per-store pooled row count must equal the
    # pre-cut transaction count recomputed here from the arrays; any drift means the
    # embedding was pooled over a different row set than the controls describe
    assert (joined["n_txns_pre_cut"].to_numpy()
            == joined["pre_txn_count"].to_numpy()).all(), (
        "CONTAMINATION: embed.py pooled a different pre-cut row set than the controls "
        "counted (n_txns_pre_cut != recomputed pre_txn_count)")
    y = np.searchsorted(edges, joined["growth"].to_numpy()).astype(np.int64)  # 0/1/2

    X_ctrl = np.column_stack([
        np.log1p(joined["pre_txn_count"].to_numpy()),
        np.log1p(joined["pre_distinct_households"].to_numpy()),
        joined["pre_mean_basket_value"].to_numpy(),
    ]).astype(np.float32)
    X_emb = np.hstack([X_ctrl, joined.select(emb_cols).to_numpy().astype(np.float32)])

    # ---- store-disjoint split (one row per store, so a plain row split IS the
    # store-disjoint split; stated rather than implied) ----
    n = joined.height
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * args.test_frac))
    te, tr = perm[:n_test], perm[n_test:]

    # fixed estimators, no early stopping: with ~100 test stores a validation carve-out
    # would be noise; deterministic settings so --check reproduces
    params = dict(objective="multiclass", num_class=N_CLASSES, metric="multi_logloss",
                  learning_rate=0.1, num_leaves=31, n_estimators=args.estimators,
                  seed=args.seed, deterministic=True, force_row_wise=True,
                  n_jobs=args.threads, verbose=-1)
    preds = {}
    for arm, X in (("control", X_ctrl), ("control_plus_embeddings", X_emb)):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X[tr], y[tr])
        preds[arm] = np.argmax(clf.predict_proba(X[te]), axis=1)
        print(f"[store_head] {arm}: {X.shape[1]} features", flush=True)

    y_te = y[te]
    labels = np.arange(N_CLASSES)

    def macro_f1(yt, yp):
        return float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))

    p_c, p_e = preds["control"], preds["control_plus_embeddings"]
    arms = {
        "control": {
            "features": CONTROL_FEATURES,
            "accuracy": float((p_c == y_te).mean()),
            "macro_f1": macro_f1(y_te, p_c),
        },
        "control_plus_embeddings": {
            "features": CONTROL_FEATURES + [f"backbone store embedding ({len(emb_cols)}d)"],
            "accuracy": float((p_e == y_te).mean()),
            "macro_f1": macro_f1(y_te, p_e),
        },
    }
    maj = np.bincount(y[tr], minlength=N_CLASSES).argmax()
    majority_floor = {
        "accuracy": float((y_te == maj).mean()),
        "macro_f1": macro_f1(y_te, np.full_like(y_te, maj)),
        "note": "predict the most frequent train tercile for every store",
    }

    fns = [
        lambda i: macro_f1(y_te[i], p_e[i]) - macro_f1(y_te[i], p_c[i]),
        lambda i: (p_e[i] == y_te[i]).mean() - (p_c[i] == y_te[i]).mean(),
    ]
    # one store per row, so the store-clustered bootstrap is a row bootstrap; the
    # machinery is the shipped paired_cluster_bootstrap with singleton clusters
    cis = paired_cluster_bootstrap(np.arange(len(y_te)), fns, args.bootstrap, args.seed)
    delta = {
        "macro_f1": arms["control_plus_embeddings"]["macro_f1"] - arms["control"]["macro_f1"],
        "macro_f1_ci": cis[0],
        "accuracy": arms["control_plus_embeddings"]["accuracy"] - arms["control"]["accuracy"],
        "accuracy_ci": cis[1],
    }

    lo, hi = delta["macro_f1_ci"]
    if lo > 0:
        sentence = (
            "On this corpus the backbone's store embeddings add measurable signal over the "
            f"counts-only control for post-cut sales growth (macro-F1 delta "
            f"{delta['macro_f1']:+.4f}, 95% interval [{lo:.4f}, {hi:.4f}] above zero).")
        call = "embeddings_add_signal"
    elif hi < 0:
        sentence = (
            "On this corpus the counts-only control beats control plus backbone store "
            f"embeddings for post-cut sales growth (macro-F1 delta {delta['macro_f1']:+.4f}, "
            f"95% interval [{lo:.4f}, {hi:.4f}] below zero). Per the pre-registration this "
            "ships as obtained.")
        call = "embeddings_hurt"
    else:
        sentence = (
            "On this corpus the backbone's store embeddings do not separate from the "
            f"counts-only control for post-cut sales growth (macro-F1 delta "
            f"{delta['macro_f1']:+.4f}, 95% interval [{lo:.4f}, {hi:.4f}] spans zero). "
            "Per the pre-registration this ships as a null.")
        call = "null"

    return {
        "seed": args.seed,
        "backbone_seed": args.backbone_seed,
        "versions": versions_dict(),
        "generated_by": "scripts/fm/cj_store_head.py --check-able",
        "data_sources": meta["data_sources"] + [
            {"name": "backbone store embeddings (pre-cut pooled, embed.py merchant path)",
             "url": str(args.emb), "sha256": sha256_file(args.emb)},
        ],
        "labels": ["real", "predictive not causal"],
        "what_this_is": "a store-level head consuming backbone store embeddings against a "
                        "real store outcome through the shared store key; the honest "
                        "question is whether embeddings add anything over counts",
        "outcome": {
            "definition": "post-cut SALES_VALUE per post day divided by pre-cut "
                          "SALES_VALUE per pre day, cut at terciles of eligible stores",
            "classes": ["low growth", "mid growth", "high growth"],
            "tercile_edges": [float(e) for e in edges],
            "pre_days": n_pre_days, "post_days": n_post_days,
            "why_growth_not_volume": "raw post-cut volume is mostly pre-cut size restated; "
                                     "the counts control would win by construction without "
                                     "answering the embedding question",
        },
        "universe": {
            **exclusions,
            "eligibility": f"pre-cut rows >= {args.min_pre_rows} and pre-cut sales > 0; "
                           "the NA store key and stores with no pre-cut rows are counted "
                           "separately above, not folded into a floor number",
            "n_train": int(len(tr)), "n_test": int(len(y_te)),
            "split": "store-disjoint (one row per store; the row split is the store split)",
        },
        "arms": arms,
        "majority_floor": majority_floor,
        "delta": delta,
        "bootstrap": {"method": "store-clustered paired bootstrap (singleton clusters = "
                                "row bootstrap; shipped machinery)",
                      "B": args.bootstrap, "ci": "percentile 95%"},
        "scale_caveat": "a few hundred stores; intervals are wide because the corpus is "
                        "small, and the pre-registration says the result ships either way",
        "required_sentence": sentence,
        "call": call,
        "check": {"command": "python3 scripts/fm/cj_store_head.py ... --check",
                  "tolerance": args.check_tol,
                  "note": "rerun with the same --threads as the producing run; "
                          "deterministic LightGBM settings are on"},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--emb", required=True, help="store embeddings parquet from cj_run embed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone-seed", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--min-pre-rows", type=int, default=MIN_PRE_ROWS)
    ap.add_argument("--estimators", type=int, default=200)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare every numeric leaf; exit 0/5")
    ap.add_argument("--check-tol", type=float, default=1e-6)
    args = ap.parse_args()

    fresh = build(args)
    if args.check:
        from ladder_eval import compare

        stored = json.loads(Path(args.out).read_text())
        return compare(fresh, stored, args.check_tol)
    atomic_write_json(args.out, fresh)
    print(f"[store_head] call={fresh['call']}")
    print(f"[store_head] {fresh['required_sentence']}")
    print(f"[store_head] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
