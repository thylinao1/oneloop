"""cj_offer_head.py - household-level offers demonstration on the Complete Journey corpus.

The shipped offers head never consumed the backbone (limitation L-B). This head predicts
whether a household redeems ANY campaign coupon in the POST-cut window from its PRE-cut
backbone embedding, against a counts-plus-demographics control.

PREDICTIVE, NOT CAUSAL, stated up front: Complete Journey campaigns were TARGETED by the
retailer, not randomized, so "households that redeem look like X" says nothing about
what an offer causes. The causal offers result in this project remains the Criteo
randomized-uplift exhibit; this head only tests whether backbone embeddings carry
predictive signal about a real redemption outcome at the household level.

Leakage hard-fails implemented HERE:
  - household embeddings are pooled over PRE-cut transactions only (stage emb below;
    the full-history pooling ladder_eval uses for its permissive rung is exactly what
    this stage refuses to do)
  - every control feature is computed from PRE-cut rows / PRE-cut redemptions only;
    demographics are static attributes
  - the label uses POST-cut redemptions only (coupon_redempt DAY >= cut_day)
  - household-disjoint split (one row per household; split sizes recorded because the
    pool is ~2,500 households)
  - COUPON_DISC never entered the backbone vocab (prep_cj.py hard-fails on it), so the
    embedding cannot have read the outcome family at pretraining time

Stages: --stage emb  (torch, CPU is fine at this scale: pre-cut pooled household
                      embeddings from the frozen checkpoint -> .npz)
        --stage run  (LightGBM arms + household-clustered bootstrap -> results JSON;
                      --check recomputes and compares numeric leaves at 1e-6)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm  # noqa: F401  MUST load before torch (seed_everything and the emb
                 # stage import torch; macOS libomp coexists only if LightGBM's loads
                 # first)
import numpy as np
import polars as pl

from common import atomic_write_json, seed_everything, versions_dict
from cj_common import HH_DEMOGRAPHIC_COLUMNS_EXPECTED, load_prep_cj, sha256_file
from transfer_eval import paired_cluster_bootstrap

# 2023 re-release anonymized the demographic columns: classification_1..5 plus
# HOMEOWNER_DESC and KID_CATEGORY_DESC. There are no AGE_DESC / INCOME_DESC columns.
DEMOGRAPHIC_FEATURES = ["classification_1", "classification_2", "classification_3",
                        "HOMEOWNER_DESC", "classification_5", "classification_4",
                        "KID_CATEGORY_DESC"]
CONTROL_COUNT_FEATURES = ["log1p_pre_txn_count", "log1p_pre_distinct_baskets",
                          "log1p_pre_total_sales", "pre_distinct_stores",
                          "log1p_pre_coupon_rows", "log1p_pre_redemptions"]


# ---------------------------------------------------------------- stage emb ----

def stage_emb(args):
    """Pre-cut pooled per-household embeddings: mean of the frozen backbone's
    per-transaction encodings over each household's PRE-cut sequence ONLY."""
    import torch

    from common import PAD_ID, user_segments
    from embed import encode_batches, load_model

    if args.threads:
        torch.set_num_threads(args.threads)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    seed_everything(args.seed)
    d = load_prep_cj(args.prep)
    device = torch.device(args.device)
    model, cfg = load_model(args.ckpt, d["meta"], device)
    W = cfg["window"]

    pre = d["ts"] < d["meta"]["cut_ts"]  # the hard leakage guard of this stage
    tokens, user = d["tokens"][pre], d["user"][pre]
    seg_starts, seg_ends = user_segments(user)
    starts, ends = [], []
    for s, e in zip(seg_starts, seg_ends):
        st = np.arange(s, e, W, dtype=np.int64)
        starts.append(st)
        ends.append(np.minimum(st + W, e))
    starts, ends = np.concatenate(starts), np.concatenate(ends)

    n_users = int(d["user"].max()) + 1
    dsum = np.zeros((n_users, model.d_model), dtype=np.float64)
    dcnt = np.zeros(n_users, dtype=np.int64)
    F_ = tokens.shape[1]
    chunk = 4096
    for c0 in range(0, len(starts), chunk):
        cs, ce = starts[c0:c0 + chunk], ends[c0:c0 + chunk]
        wins = np.full((len(cs), W, F_), PAD_ID, dtype=np.int64)
        rows = np.full((len(cs), W), -1, dtype=np.int64)
        for j, (s, e) in enumerate(zip(cs, ce)):
            wins[j, : e - s] = tokens[s:e]
            rows[j, : e - s] = np.arange(s, e)
        for sl, h in encode_batches(model, wins, ce - cs, device, args.batch_size):
            r = rows[sl]
            valid = r >= 0
            np.add.at(dsum, user[r[valid]], h.numpy()[valid])
            np.add.at(dcnt, user[r[valid]], 1)
        if c0 % (chunk * 20) == 0:
            print(f"[offer_emb] pre-cut pooling {c0}/{len(starts)} window-chunks", flush=True)

    emb = np.zeros_like(dsum, dtype=np.float32)
    ok = dcnt > 0
    emb[ok] = (dsum[ok] / dcnt[ok, None]).astype(np.float32)
    np.savez(args.out, emb=emb, count=dcnt)
    print(f"[offer_emb] {int(ok.sum())}/{n_users} households pooled over "
          f"{int(dcnt.sum()):,} PRE-cut transactions -> {args.out}", flush=True)


# ---------------------------------------------------------------- stage run ----

def build(args) -> dict:
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    seed_everything(args.seed)
    d = load_prep_cj(args.prep)
    meta = d["meta"]
    pre = d["ts"] < meta["cut_ts"]
    cut_day = int(meta["cut_day"])
    basket = np.load(Path(args.prep) / "basket.npy")
    hh_keys = pl.read_parquet(Path(args.prep) / "household_keys.parquet")

    # ---- per-household PRE-cut count controls ----
    df = pl.DataFrame({
        "user": d["user"], "basket": basket, "store": d["merchant"],
        "amount": d["amount"].astype(np.float64),
        "coupon_row": d["fraud"].astype(np.int64),  # COUPON_DISC<0 row label (see meta)
        "pre": pre,
    })
    ctrl = (
        df.filter(pl.col("pre"))
        .group_by("user")
        .agg(
            pl.len().alias("pre_txn_count"),
            pl.col("basket").n_unique().alias("pre_distinct_baskets"),
            pl.col("amount").sum().alias("pre_total_sales"),
            pl.col("store").n_unique().alias("pre_distinct_stores"),
            pl.col("coupon_row").sum().alias("pre_coupon_rows"),
        )
    )

    # ---- redemptions: pre-cut count (control) and post-cut any (LABEL) ----
    red = pl.read_parquet(args.redemptions) if str(args.redemptions).endswith(".parquet") \
        else pl.read_csv(args.redemptions)
    assert {"household_key", "DAY"} <= set(red.columns), "coupon_redempt schema drift"
    assert hh_keys["user"].n_unique() == hh_keys.height, \
        "household_keys.parquet has duplicate user rows"
    assert hh_keys["household_key"].n_unique() == hh_keys.height, \
        "household_keys.parquet has duplicate household_key rows"
    red = red.join(hh_keys, on="household_key", how="inner")
    pre_red = (red.filter(pl.col("DAY") < cut_day)
               .group_by("user").agg(pl.len().alias("pre_redemptions")))
    post_red_users = set(red.filter(pl.col("DAY") >= cut_day)["user"].to_list())

    # ---- demographics (anonymized attributes; 801 of ~2,500 households have them) ----
    demo = pl.read_parquet(args.demographics) if str(args.demographics).endswith(".parquet") \
        else pl.read_csv(args.demographics)
    missing = [c for c in HH_DEMOGRAPHIC_COLUMNS_EXPECTED if c not in demo.columns]
    assert not missing, f"hh_demographic schema drift, missing: {missing}"
    demo = demo.join(hh_keys, on="household_key", how="inner").select(
        "user", *[pl.col(c).cast(pl.Utf8).fill_null("NA") for c in DEMOGRAPHIC_FEATURES]
    )
    # uniqueness guards: a duplicate on either side of these joins would silently
    # duplicate households into the design matrix
    assert demo["user"].n_unique() == demo.height, \
        "hh_demographic join produced duplicate households"
    assert pre_red["user"].n_unique() == pre_red.height, \
        "pre-cut redemption aggregate has duplicate households"

    base = (
        ctrl.join(pre_red, on="user", how="left")
        .join(demo, on="user", how="left")
        .with_columns(
            pl.col("pre_redemptions").fill_null(0),
            *[pl.col(c).fill_null("NA") for c in DEMOGRAPHIC_FEATURES],
        )
        .sort("user")
    )
    assert base.height == ctrl.height, \
        "joins changed the household row count (expected one row per household)"
    assert base["user"].n_unique() == base.height, "duplicate households in the design matrix"
    users = base["user"].to_numpy()
    y = np.isin(users, list(post_red_users)).astype(np.int64)

    demo_codes = np.column_stack([
        base[c].cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy()
        for c in DEMOGRAPHIC_FEATURES
    ]).astype(np.float32)
    X_ctrl = np.column_stack([
        np.log1p(base["pre_txn_count"].to_numpy()),
        np.log1p(base["pre_distinct_baskets"].to_numpy()),
        np.log1p(base["pre_total_sales"].to_numpy().clip(0)),
        base["pre_distinct_stores"].to_numpy(),
        np.log1p(base["pre_coupon_rows"].to_numpy()),
        np.log1p(base["pre_redemptions"].to_numpy()),
        demo_codes,
    ]).astype(np.float32)

    ue = np.load(args.emb)
    emb_all, cnt_all = ue["emb"], ue["count"]
    emb = emb_all[users]
    has_emb = (cnt_all[users] > 0).astype(np.float32)[:, None]
    X_emb = np.hstack([X_ctrl, emb, has_emb])

    # ---- household-disjoint split (one row per household) ----
    n = len(users)
    # REPORTED count from prep metadata, never from max+1: user.npy holds the raw
    # 1-based household_key, so int(user.max()) + 1 is an array SIZE (slot 0 unused),
    # not a household count
    n_households_total = int(meta["cj"]["n_households"])
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * args.test_frac))
    te, tr = perm[:n_test], perm[n_test:]
    y_te = y[te]
    n_pos_te, n_neg_te = int(y_te.sum()), int(len(y_te) - y_te.sum())

    def unscorable_json(sentence: str, call: str, extra: dict | None = None) -> dict:
        out = {
            "seed": args.seed,
            "backbone_seed": args.backbone_seed,
            "versions": versions_dict(),
            "generated_by": "scripts/fm/cj_offer_head.py --check-able",
            "data_sources": meta["data_sources"],
            "labels": ["real", "predictive not causal", "unscorable"],
            "label": {"definition": f"household has >=1 coupon_redempt row with DAY >= "
                                    f"{cut_day} (the corpus cut)",
                      "n_households": int(n), "n_positive": int(y.sum()),
                      "n_test_positive": n_pos_te, "n_test_negative": n_neg_te},
            "universe": {"n_train": int(len(tr)), "n_test": int(len(te)),
                         "n_households_total": n_households_total,
                         "n_excluded_no_pre_rows": int(n_households_total - n)},
            "call": call,
            "required_sentence": sentence,
            "check": {"command": "python3 scripts/fm/cj_offer_head.py --stage run ... --check",
                      "tolerance": args.check_tol},
        }
        if extra:
            out.update(extra)
        return out

    # pre-registered: too few label examples on either side is REPORTED as the finding,
    # not tuned around and not crashed on (floor --min-test-pos, default 5, set before
    # any label was computed)
    if min(n_pos_te, n_neg_te) < args.min_test_pos:
        return unscorable_json(
            "The test split holds too few households on one side of the label to score "
            f"this head ({n_pos_te} positive and {n_neg_te} negative among "
            f"{int(len(y_te))} test households, floor {args.min_test_pos}). Per the "
            "pre-registration this ships as the finding; no window or split is adjusted "
            "to manufacture a scoreable label.",
            "unscorable")

    params = dict(objective="binary", metric="average_precision", learning_rate=0.05,
                  num_leaves=31, n_estimators=args.estimators, seed=args.seed,
                  deterministic=True, force_row_wise=True, n_jobs=args.threads,
                  verbose=-1)
    scores = {}
    for arm, X in (("control", X_ctrl), ("control_plus_embeddings", X_emb)):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X[tr], y[tr])
        scores[arm] = clf.predict_proba(X[te])[:, 1]
        print(f"[offer_head] {arm}: {X.shape[1]} features", flush=True)

    s_c, s_e = scores["control"], scores["control_plus_embeddings"]
    arms = {
        "control": {
            "features": CONTROL_COUNT_FEATURES + [f"{c} (anonymized categorical)"
                                                  for c in DEMOGRAPHIC_FEATURES],
            "auc": float(roc_auc_score(y_te, s_c)),
            "prauc": float(average_precision_score(y_te, s_c)),
        },
        "control_plus_embeddings": {
            "features": "control features + pre-cut pooled backbone household embedding "
                        f"({emb.shape[1]}d) + has_history flag",
            "auc": float(roc_auc_score(y_te, s_e)),
            "prauc": float(average_precision_score(y_te, s_e)),
        },
    }
    fns = [
        lambda i: roc_auc_score(y_te[i], s_e[i]) - roc_auc_score(y_te[i], s_c[i]),
        lambda i: average_precision_score(y_te[i], s_e[i])
        - average_precision_score(y_te[i], s_c[i]),
    ]
    # one household per row: the household-clustered bootstrap is a row bootstrap on
    # households, run through the shipped machinery
    cis = paired_cluster_bootstrap(np.arange(len(y_te)), fns, args.bootstrap, args.seed)
    if np.isnan(np.asarray(cis, dtype=np.float64)).any():
        # every (or nearly every) resample drew a single-class test set, so the
        # percentile interval does not exist; ship that instead of NaN CIs
        return unscorable_json(
            "The household bootstrap degenerated (resamples with a single label class; "
            f"{n_pos_te} test positives), so no interval exists for the delta. Per the "
            "pre-registration this ships as the finding.",
            "unscorable_bootstrap",
            extra={"arms": arms,
                   "delta_point_estimates": {
                       "auc": arms["control_plus_embeddings"]["auc"] - arms["control"]["auc"],
                       "prauc": arms["control_plus_embeddings"]["prauc"]
                       - arms["control"]["prauc"]},
                   "bootstrap": {"method": "household-clustered paired bootstrap",
                                 "B": args.bootstrap, "ci": "degenerate (see call)"}})
    delta = {
        "auc": arms["control_plus_embeddings"]["auc"] - arms["control"]["auc"],
        "auc_ci": cis[0],
        "prauc": arms["control_plus_embeddings"]["prauc"] - arms["control"]["prauc"],
        "prauc_ci": cis[1],
    }

    lo, hi = delta["auc_ci"]
    if lo > 0:
        sentence = (
            "On this corpus the backbone's pre-cut household embeddings add measurable "
            "signal over the counts-plus-demographics control for post-cut coupon "
            f"redemption (AUC delta {delta['auc']:+.4f}, 95% interval [{lo:.4f}, {hi:.4f}] "
            "above zero). This is a predictive statement about targeted campaigns, not a "
            "causal one.")
        call = "embeddings_add_signal"
    elif hi < 0:
        sentence = (
            "On this corpus the counts-plus-demographics control beats control plus "
            f"backbone household embeddings for post-cut coupon redemption (AUC delta "
            f"{delta['auc']:+.4f}, 95% interval [{lo:.4f}, {hi:.4f}] below zero). Per the "
            "pre-registration this ships as obtained.")
        call = "embeddings_hurt"
    else:
        sentence = (
            "On this corpus the backbone's pre-cut household embeddings do not separate "
            "from the counts-plus-demographics control for post-cut coupon redemption "
            f"(AUC delta {delta['auc']:+.4f}, 95% interval [{lo:.4f}, {hi:.4f}] spans "
            "zero). Per the pre-registration this ships as a null.")
        call = "null"

    return {
        "seed": args.seed,
        "backbone_seed": args.backbone_seed,
        "versions": versions_dict(),
        "generated_by": "scripts/fm/cj_offer_head.py --check-able",
        "data_sources": meta["data_sources"] + [
            {"name": "coupon_redempt (campaign coupon redemptions)",
             "url": str(args.redemptions), "sha256": sha256_file(args.redemptions)},
            {"name": "hh_demographic (anonymized household attributes, 2023 re-release)",
             "url": str(args.demographics), "sha256": sha256_file(args.demographics)},
        ],
        "labels": ["real", "predictive not causal"],
        "what_this_is": "a household-level offers head consuming pre-cut backbone "
                        "household embeddings against a real redemption outcome; the "
                        "honest question is whether embeddings add anything over counts "
                        "plus demographics",
        "not_causal": "campaigns in this corpus were targeted by the retailer, not "
                      "randomized; nothing here estimates what an offer causes. The "
                      "causal offers exhibit in this project remains the Criteo "
                      "randomized-uplift result.",
        "label": {
            "definition": f"household has >=1 coupon_redempt row with DAY >= {cut_day} "
                          "(the corpus cut)",
            "n_households": int(n),
            "n_positive": int(y.sum()),
            "n_test_positive": n_pos_te,
            "n_test_negative": n_neg_te,
        },
        "universe": {
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_households_total": n_households_total,
            "n_excluded_no_pre_rows": int(n_households_total - n),
            "split": "household-disjoint (one row per household; the row split is the "
                     "household split)",
            "n_households_with_demographics": int(demo.height),
            "demographics_note": "the 2023 re-release anonymized demographic columns to "
                                 "classification_1..5 + HOMEOWNER_DESC + "
                                 "KID_CATEGORY_DESC; households without a demographic "
                                 "row carry 'NA' codes",
        },
        "arms": arms,
        "delta": delta,
        "bootstrap": {"method": "household-clustered paired bootstrap (singleton "
                                "clusters = row bootstrap; shipped machinery)",
                      "B": args.bootstrap, "ci": "percentile 95%"},
        "scale_caveat": "~2,500 households and a rare positive; the positive counts "
                        "above are the whole story and the intervals are wide",
        "required_sentence": sentence,
        "call": call,
        "check": {"command": "python3 scripts/fm/cj_offer_head.py --stage run ... --check",
                  "tolerance": args.check_tol,
                  "note": "rerun with the same --threads as the producing run; "
                          "deterministic LightGBM settings are on"},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["emb", "run"])
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True, help="emb: .npz path | run: results json path")
    ap.add_argument("--ckpt", default="", help="emb: frozen checkpoint")
    ap.add_argument("--emb", default="", help="run: .npz from the emb stage")
    ap.add_argument("--redemptions", default="", help="run: coupon_redempt csv/parquet")
    ap.add_argument("--demographics", default="", help="run: hh_demographic csv/parquet")
    ap.add_argument("--backbone-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--min-test-pos", type=int, default=5,
                    help="run: fewer than this many test households on either label side "
                         "makes the head unscorable (reported, never tuned around)")
    ap.add_argument("--estimators", type=int, default=300)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--check", action="store_true",
                    help="run stage: recompute and compare numeric leaves; exit 0/5")
    ap.add_argument("--check-tol", type=float, default=1e-6)
    args = ap.parse_args()

    if args.stage == "emb":
        assert args.ckpt, "--ckpt required for the emb stage"
        stage_emb(args)
        return 0

    for req in ("emb", "redemptions", "demographics"):
        assert getattr(args, req), f"--{req} required for the run stage"
    assert args.backbone_seed, "--backbone-seed required for the run stage"
    fresh = build(args)
    if args.check:
        from ladder_eval import compare

        stored = json.loads(Path(args.out).read_text())
        return compare(fresh, stored, args.check_tol)
    atomic_write_json(args.out, fresh)
    print(f"[offer_head] call={fresh['call']}")
    print(f"[offer_head] {fresh['required_sentence']}")
    print(f"[offer_head] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
