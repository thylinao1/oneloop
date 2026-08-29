"""ladder_eval.py: the leakage ladder. Evaluation-only, one frozen checkpoint.

Published transaction-foundation-model demonstrations report large downstream lifts.
Ours is small. This script answers "is that the model or the protocol?" by measurement:
the SAME frozen backbone is evaluated on the SAME two downstream tasks under four
progressively stricter evaluation protocols, changing ONE guard at a time.

  L0  permissive (the shape common in published demonstrations)
        - temporal-only split: post-cut rows of ALL accounts, earliest 8/11 train,
          latest 3/11 test, so the same accounts appear on both sides
        - baseline WITHOUT per-entity temporal aggregates (raw fields only)
        - FULL-HISTORY pooled account embedding: one vector per account over its
          entire sequence, so it can see the scored transaction and everything after
  L1  = L0 + entity-disjoint split by account
  L2  = L1 + as-of prefix-only embeddings (strictly before the scored transaction)
  L3  = L2 + baseline equipped with per-entity temporal aggregates
        == THE SHIPPED PROTOCOL. Asserted to reproduce results/backbone_transfer.json.

The two pretrain-side guards recorded in backbone_transfer.json (label columns excluded
from the pretraining vocabulary; account/card identifiers never in the vocabulary) and
the corpus time truncation stay ON at every rung. Only the EVALUATION protocol moves.

Disclosed: the merchant-side embedding that the fraud arm consumes is the pre-cut pooled
one at every rung. Only the ACCOUNT-side embedding varies, so L0 is a LOWER bound on how
much a fully permissive protocol inflates, not an upper bound.

Stages
  --stage user_emb : GPU pass; full-history pooled per-account embeddings -> .npz
  --stage run      : CPU; four rungs -> results/ladder.json
  --stage run --check : recompute and compare every numeric leaf at 1e-6; exit 0/5
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from common import FIELDS, load_prep, seed_everything, atomic_write_json, versions_dict
from transfer_eval import (VAL_USER_FRAC, cap_rows, freq_encode, history_features,
                           paired_cluster_bootstrap)

# L0 temporal split keeps the shipped 800k/300k train:test ratio (8:11 of the post-cut pool).
L0_TRAIN_FRAC = 8.0 / 11.0

# history_features column order used by build_matrices below.
HIST_COLS = ["hist_count", "amt_mean", "amt_std", "amt_max", "last_amount",
             "recency_log", "since_first_log", "freq_prev_mcc", "freq_prev_city", "prev_mcc"]
# Columns that summarize an account's history = the per-entity temporal aggregates.
HIST_AGG_IDX = [0, 1, 2, 3, 5, 6]
HIST_RAW_IDX = [4, 7, 8, 9]
# fraud current-transaction block; index 1 (the amount z-score) is aggregate-derived.
CUR_COLS = ["amount", "amount_z", "hour", "use_chip", "freq_mcc", "freq_city", "freq_state", "mcc"]
CUR_RAW_IDX = [0, 2, 3, 4, 5, 6, 7]

RUNGS = [
    ("L0", "permissive", dict(entity_disjoint_split=False, as_of_embeddings=False,
                              baseline_aggregates=False)),
    ("L1", "entity-disjoint split", dict(entity_disjoint_split=True, as_of_embeddings=False,
                                         baseline_aggregates=False)),
    ("L2", "as-of embeddings", dict(entity_disjoint_split=True, as_of_embeddings=True,
                                    baseline_aggregates=False)),
    ("L3", "aggregate-equipped baseline (shipped)", dict(entity_disjoint_split=True,
                                                         as_of_embeddings=True,
                                                         baseline_aggregates=True)),
]

SHIPPED_KEYS = {
    "fraud": ["baseline_auc", "baseline_prauc", "with_emb_auc", "with_emb_prauc"],
    "next_mcc": ["baseline_top1", "baseline_top5", "with_emb_top1", "with_emb_top5"],
}


# ------------------------------------------------------------ stage user_emb ----

def stage_user_emb(args):
    """Full-history pooled per-account embedding: mean of the backbone's per-transaction
    encodings over the account's ENTIRE sequence (pre-cut AND post-cut rows).

    This is the permissive construction the ladder removes at L2. It deliberately lets an
    account's embedding see the scored transaction and every transaction after it.
    """
    import torch

    from common import PAD_ID, user_segments
    from embed import encode_batches, load_model

    if args.threads:
        torch.set_num_threads(args.threads)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    seed_everything(args.seed)
    d = load_prep(args.prep)
    device = torch.device(args.device)
    model, cfg = load_model(args.ckpt, d["meta"], device)
    W = cfg["window"]

    tokens, user = d["tokens"], d["user"]
    seg_starts, seg_ends = user_segments(user)
    starts, ends = [], []
    for s, e in zip(seg_starts, seg_ends):
        st = np.arange(s, e, W, dtype=np.int64)
        starts.append(st)
        ends.append(np.minimum(st + W, e))
    starts, ends = np.concatenate(starts), np.concatenate(ends)

    n_users = int(user.max()) + 1
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
            print(f"[user_emb] full-history pooling {c0}/{len(starts)} window-chunks", flush=True)

    emb = np.zeros_like(dsum, dtype=np.float32)
    ok = dcnt > 0
    emb[ok] = (dsum[ok] / dcnt[ok, None]).astype(np.float32)
    np.savez(args.out, emb=emb, count=dcnt)
    print(f"[user_emb] {int(ok.sum())}/{n_users} accounts pooled over "
          f"{int(dcnt.sum()):,} transactions -> {args.out}", flush=True)


# ------------------------------------------------------------------ features ----

def build_matrices(d, hf, rows, user_vec, merch_emb, merch_has, freq, task, agg):
    """(X_base, X_emb) for `rows` (sorted ascending). `agg` toggles the aggregate block."""
    sub = hf.with_row_index("_r").filter(pl.col("_r").is_in(pl.Series(rows))).sort("_r")
    assert sub.height == len(rows), "row selection mismatch"
    hist = np.column_stack([
        sub["hist_count"].fill_null(0).to_numpy(),
        sub["amt_mean"].fill_null(0.0).to_numpy(),
        sub["amt_std"].fill_null(0.0).to_numpy(),
        sub["amt_max"].fill_null(0.0).to_numpy(),
        sub["last_amount"].fill_null(0.0).to_numpy(),
        np.log1p(sub["recency_s"].fill_null(-1).to_numpy().clip(0)),
        np.log1p(sub["since_first_s"].fill_null(0).to_numpy()),
        freq_encode(sub["prev_mcc"].fill_null(0).to_numpy(), freq["mcc"]),
        freq_encode(sub["prev_city"].fill_null(0).to_numpy(), freq["city"]),
        sub["prev_mcc"].fill_null(0).to_numpy(),
    ]).astype(np.float32)
    hist = hist if agg else hist[:, HIST_RAW_IDX]
    if task == "fraud":
        amt = sub["amount"].to_numpy()
        mean = sub["amt_mean"].fill_null(0.0).to_numpy()
        std = sub["amt_std"].fill_null(0.0).to_numpy()
        z = (amt - mean) / np.where(std > 1e-6, std, 1.0)
        cur = np.column_stack([
            amt, z,
            sub["hour_tok"].to_numpy(), sub["chip_tok"].to_numpy(),
            freq_encode(sub["mcc_tok"].to_numpy(), freq["mcc"]),
            freq_encode(sub["city_tok"].to_numpy(), freq["city"]),
            freq_encode(sub["state_tok"].to_numpy(), freq["state"]),
            sub["mcc_tok"].to_numpy(),
        ]).astype(np.float32)
        cur = cur if agg else cur[:, CUR_RAW_IDX]
        X_base = np.hstack([hist, cur])
        me = merch_emb[d["merchant"][rows]]
        mh = merch_has[d["merchant"][rows]][:, None].astype(np.float32)
        X_emb = np.hstack([X_base, user_vec, me, mh])
    else:  # next_mcc: the scored transaction's own fields are the answer, never inputs
        X_base = hist
        X_emb = np.hstack([X_base, user_vec])
    return X_base, X_emb


def feature_names(task, agg):
    hist = HIST_COLS if agg else [HIST_COLS[i] for i in HIST_RAW_IDX]
    if task != "fraud":
        return hist
    cur = CUR_COLS if agg else [CUR_COLS[i] for i in CUR_RAW_IDX]
    return hist + cur


# ---------------------------------------------------------------------- rungs ----

def l0_split(d, seed, max_train, max_test):
    """Temporal-only split of the post-cut pool: the same accounts on both sides."""
    post = np.flatnonzero(d["ts"] >= d["meta"]["cut_ts"])
    order = np.argsort(d["ts"][post], kind="stable")
    n_tr = int(round(len(post) * L0_TRAIN_FRAC))
    rng = np.random.default_rng(seed)
    train_rows = cap_rows(post[order[:n_tr]], d["fraud"], max_train, rng)
    test_rows = cap_rows(post[order[n_tr:]], d["fraud"], max_test, rng)
    return train_rows, test_rows


def score_rung(d, hf, train_rows, test_rows, emb_train, emb_test, merch_emb, merch_has,
               freq, args, agg):
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    meta = d["meta"]
    rng = np.random.default_rng(args.seed)
    tr_users = d["user"][train_rows]
    uniq_tr = np.unique(tr_users)
    val_users = set(rng.permutation(uniq_tr)[: max(1, int(len(uniq_tr) * VAL_USER_FRAC))].tolist())
    val_mask = np.isin(tr_users, list(val_users))
    u_te = d["user"][test_rows]
    out = {}
    # per-row scores kept so a LATER rung can be compared to this one on the SAME test rows
    # with a paired bootstrap (a guard's price needs an interval, not just two point estimates)
    keep = {}

    for task in ("fraud", "next_mcc"):
        Xb_tr, Xe_tr = build_matrices(d, hf, train_rows, emb_train, merch_emb, merch_has,
                                      freq, task, agg)
        Xb_te, Xe_te = build_matrices(d, hf, test_rows, emb_test, merch_emb, merch_has,
                                      freq, task, agg)
        if task == "fraud":
            y_tr, y_te = d["fraud"][train_rows], d["fraud"][test_rows]
            params = dict(objective="binary", metric="average_precision", learning_rate=0.05,
                          num_leaves=63, n_estimators=args.fraud_estimators, seed=args.seed,
                          n_jobs=args.threads or -1, verbose=-1)
        else:
            y_tr, y_te = d["mcc_class"][train_rows], d["mcc_class"][test_rows]
            params = dict(objective="multiclass", num_class=meta["n_mcc_classes"],
                          metric="multi_logloss", learning_rate=0.1, num_leaves=31,
                          n_estimators=args.mcc_estimators, seed=args.seed,
                          n_jobs=args.threads or -1, verbose=-1)
        preds = {}
        for arm, (Xtr, Xte) in (("baseline", (Xb_tr, Xb_te)), ("with_emb", (Xe_tr, Xe_te))):
            clf = lgb.LGBMClassifier(**params)
            clf.fit(Xtr[~val_mask], y_tr[~val_mask],
                    eval_set=[(Xtr[val_mask], y_tr[val_mask])],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
            preds[arm] = clf.predict_proba(Xte)
            print(f"[rung] {task}/{arm}: {clf.best_iteration_ or params['n_estimators']} iters, "
                  f"{Xtr.shape[1]} features", flush=True)
            del clf
        if task == "fraud":
            s_b, s_e = preds["baseline"][:, 1], preds["with_emb"][:, 1]
            res = {
                "baseline_auc": float(roc_auc_score(y_te, s_b)),
                "baseline_prauc": float(average_precision_score(y_te, s_b)),
                "with_emb_auc": float(roc_auc_score(y_te, s_e)),
                "with_emb_prauc": float(average_precision_score(y_te, s_e)),
            }
            res["delta_auc"] = res["with_emb_auc"] - res["baseline_auc"]
            res["delta_prauc"] = res["with_emb_prauc"] - res["baseline_prauc"]
            fns = [
                lambda i: roc_auc_score(y_te[i], s_e[i]) - roc_auc_score(y_te[i], s_b[i]),
                lambda i: average_precision_score(y_te[i], s_e[i])
                - average_precision_score(y_te[i], s_b[i]),
            ]
            res["delta_auc_ci"], res["delta_prauc_ci"] = paired_cluster_bootstrap(
                u_te, fns, args.bootstrap, args.seed)
            res["n_test_pos"] = int(y_te.sum())
            res["test_pos_rate"] = float(y_te.mean())
            keep["fraud"] = {"s_b": s_b.astype(np.float32), "s_e": s_e.astype(np.float32),
                             "y": y_te}
        else:
            def topk(p, y, k):
                return (np.argsort(-p, axis=1)[:, :k] == y[:, None]).any(axis=1).astype(np.float64)

            hit1_b, hit5_b = topk(preds["baseline"], y_te, 1), topk(preds["baseline"], y_te, 5)
            hit1_e, hit5_e = topk(preds["with_emb"], y_te, 1), topk(preds["with_emb"], y_te, 5)
            res = {
                "baseline_top1": float(hit1_b.mean()), "baseline_top5": float(hit5_b.mean()),
                "with_emb_top1": float(hit1_e.mean()), "with_emb_top5": float(hit5_e.mean()),
                "n_classes": meta["n_mcc_classes"],
                "features": "history-only (scored txn fields excluded)",
            }
            res["delta_top1"] = res["with_emb_top1"] - res["baseline_top1"]
            res["delta_top5"] = res["with_emb_top5"] - res["baseline_top5"]
            fns = [lambda i: hit1_e[i].mean() - hit1_b[i].mean(),
                   lambda i: hit5_e[i].mean() - hit5_b[i].mean()]
            res["delta_top1_ci"], res["delta_top5_ci"] = paired_cluster_bootstrap(
                u_te, fns, args.bootstrap, args.seed)
            keep["next_mcc"] = {"hit1_b": hit1_b.astype(np.uint8), "hit1_e": hit1_e.astype(np.uint8),
                                "hit5_b": hit5_b.astype(np.uint8), "hit5_e": hit5_e.astype(np.uint8)}
        res["n_train"], res["n_test"] = int(len(train_rows)), int(len(test_rows))
        res["n_test_users"] = int(len(np.unique(u_te)))
        res["n_features_baseline"] = int(Xb_tr.shape[1])
        res["n_features_with_emb"] = int(Xe_tr.shape[1])
        res["baseline_features"] = ", ".join(feature_names(task, agg))
        out[task] = res
        print(f"[rung] {task}: {res}", flush=True)
        del Xb_tr, Xe_tr, Xb_te, Xe_te, preds
        gc.collect()
    return out, keep


def guard_price_cis(kept, u_te, args):
    """Paired 95% intervals on a guard's PRICE (delta_after minus delta_before).

    Only defined between rungs scored on the SAME test rows, which is L1 vs L2 and L2 vs L3.
    The L0 to L1 step changes which rows are scored, so no paired interval exists there and
    the results file says so rather than printing a bare difference as if it were measured.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = []
    for a, b, guard in (("L1", "L2", "as-of prefix-only embeddings"),
                        ("L2", "L3", "baseline equipped with per-entity temporal aggregates")):
        ka, kb = kept[a], kept[b]
        na, nb = ka["next_mcc"], kb["next_mcc"]
        fa, fb = ka["fraud"], kb["fraud"]
        assert np.array_equal(fa["y"], fb["y"]), "paired price needs identical test rows"
        y = fa["y"]

        def topdiff(lo, hi, key_b, key_e):
            return lambda i: ((hi[key_e][i].mean() - hi[key_b][i].mean())
                              - (lo[key_e][i].mean() - lo[key_b][i].mean()))

        def scorediff(metric):
            return lambda i: ((metric(y[i], fb["s_e"][i]) - metric(y[i], fb["s_b"][i]))
                              - (metric(y[i], fa["s_e"][i]) - metric(y[i], fa["s_b"][i])))

        fns = [topdiff(na, nb, "hit1_b", "hit1_e"), topdiff(na, nb, "hit5_b", "hit5_e"),
               scorediff(roc_auc_score), scorediff(average_precision_score)]
        cis = paired_cluster_bootstrap(u_te, fns, args.bootstrap, args.seed)
        for metric, ci in zip(("next_mcc_top1", "next_mcc_top5", "fraud_auc", "fraud_prauc"), cis):
            out.append({"guard": guard, "from": a, "to": b, "metric": metric,
                        "price_ci": ci, "paired": True,
                        "note": "paired entity-clustered bootstrap on the same test rows"})
    return out


def run_ladder(args) -> dict:
    seed_everything(args.seed)
    d = load_prep(args.prep)
    meta = d["meta"]
    ev = Path(args.eval)
    sel = np.load(ev / "select.npz")
    ship_train, ship_test = sel["train_rows"], sel["test_rows"]
    if args.smoke:
        # code-path smoke only: shrinks every rung so a crash surfaces in minutes.
        # Never used for a reported number (the L3 reproduction assert is skipped).
        args.max_train, args.max_test = args.smoke, max(1, int(args.smoke * 3 / 8))
        ship_train, ship_test = ship_train[:args.max_train], ship_test[:args.max_test]
        args.shipped = ""
    test_users = set(sel["test_users"].tolist())
    eval_rows = np.load(ev / "eval_rows.npy")
    asof_all = np.load(args.asof, mmap_mode="r")
    assert len(asof_all) == len(eval_rows), "as-of embeddings misaligned with eval rows"
    pos_of = {int(r): i for i, r in enumerate(eval_rows)}
    asof_train = np.asarray(asof_all[[pos_of[int(r)] for r in ship_train]])
    asof_test = np.asarray(asof_all[[pos_of[int(r)] for r in ship_test]])
    del asof_all

    ue = np.load(args.user_emb)
    user_full, user_cnt = ue["emb"], ue["count"]
    full_dim = user_full.shape[1] + 1
    assert full_dim == asof_train.shape[1], (
        f"full-history embedding width {full_dim} != as-of width {asof_train.shape[1]}")

    def full_hist_vec(rows):
        u = d["user"][rows]
        return np.hstack([user_full[u], (user_cnt[u] > 0).astype(np.float32)[:, None]])

    mdf = pl.read_parquet(args.merchants)
    emb_cols = [c for c in mdf.columns if c.startswith("emb_")]
    n_merch = int(d["merchant"].max()) + 1
    merch_emb = np.zeros((n_merch, len(emb_cols)), dtype=np.float32)
    merch_has = np.zeros(n_merch, dtype=bool)
    mid = mdf["merchant_id"].to_numpy()
    merch_emb[mid] = mdf.select(emb_cols).to_numpy().astype(np.float32)
    merch_has[mid] = True
    del mdf
    gc.collect()

    # frequency encodings. The entity-disjoint guard also governs the fit set: with the
    # guard OFF (L0) every pre-cut row is fair game; with it ON only pre-cut rows of
    # accounts that are not held out (identical to the shipped run).
    pre = d["ts"] < meta["cut_ts"]
    fi = {f: i for i, f in enumerate(FIELDS)}

    def freq_from(mask):
        out = {}
        for name in ("mcc", "city", "state"):
            tok = d["tokens"][:, fi[name]].astype(np.int64)
            cnt = np.bincount(tok[mask], minlength=meta["vocab_sizes"][fi[name]]).astype(np.float64)
            out[name] = (cnt / max(1, cnt.sum())).astype(np.float32)
        return out

    freq_open = freq_from(pre)
    freq_disjoint = freq_from(pre & ~np.isin(d["user"], list(test_users)))

    l0_train, l0_test = l0_split(d, args.seed, args.max_train, args.max_test)
    shared = np.intersect1d(np.unique(d["user"][l0_train]), np.unique(d["user"][l0_test]))
    n_shared = int(len(shared))
    # The guard being REMOVED must verifiably be off: L0 has to leak accounts across sides.
    assert n_shared > 0, "L0 assertion failed: the temporal split did not share any account"
    for name, tr, te in (("L1/L2/L3", ship_train, ship_test),):
        ov = np.intersect1d(np.unique(d["user"][tr]), np.unique(d["user"][te]))
        assert len(ov) == 0, f"{name} assertion failed: {len(ov)} accounts shared across the split"
    print(f"[ladder] L0 shares {n_shared} accounts across train/test; shipped split shares 0",
          flush=True)

    print("[ladder] building history features...", flush=True)
    hf = history_features(d)

    rungs, kept = [], {}
    for rid, name, guards in RUNGS:
        print(f"[ladder] ===== rung {rid} ({name}) =====", flush=True)
        if guards["entity_disjoint_split"]:
            tr, te, freq = ship_train, ship_test, freq_disjoint
        else:
            tr, te, freq = l0_train, l0_test, freq_open
        if guards["as_of_embeddings"]:
            e_tr, e_te = asof_train, asof_test
        else:
            e_tr, e_te = full_hist_vec(tr), full_hist_vec(te)
        tasks, kept[rid] = score_rung(d, hf, tr, te, e_tr, e_te, merch_emb, merch_has, freq,
                                      args, guards["baseline_aggregates"])
        rungs.append({
            "rung": rid,
            "name": name,
            "guards": guards,
            "split": ("temporal-only (the same accounts appear in train and test)"
                      if not guards["entity_disjoint_split"] else "entity-disjoint by account"),
            "account_embedding": ("full-history pooled (sees the scored transaction and after)"
                                  if not guards["as_of_embeddings"]
                                  else "as-of prefix-only (strictly before the scored transaction)"),
            "baseline": ("raw fields only, no per-entity temporal aggregates"
                         if not guards["baseline_aggregates"]
                         else "raw fields plus per-entity temporal aggregates"),
            "n_shared_accounts_train_test": n_shared if not guards["entity_disjoint_split"] else 0,
            "tasks": tasks,
        })
        del e_tr, e_te
        gc.collect()

    by_id = {r["rung"]: r for r in rungs}
    deltas_by_rung = {
        "next_mcc_top1": {r["rung"]: r["tasks"]["next_mcc"]["delta_top1"] for r in rungs},
        "next_mcc_top5": {r["rung"]: r["tasks"]["next_mcc"]["delta_top5"] for r in rungs},
        "fraud_auc": {r["rung"]: r["tasks"]["fraud"]["delta_auc"] for r in rungs},
        "fraud_prauc": {r["rung"]: r["tasks"]["fraud"]["delta_prauc"] for r in rungs},
    }
    price_ci = {(e["from"], e["to"], e["metric"]): e
                for e in guard_price_cis(kept, d["user"][ship_test], args)}
    guard_prices = []
    for a, b, guard in (("L0", "L1", "entity-disjoint split by account"),
                        ("L1", "L2", "as-of prefix-only embeddings"),
                        ("L2", "L3", "baseline equipped with per-entity temporal aggregates")):
        for metric, series in deltas_by_rung.items():
            row = {"guard": guard, "from": a, "to": b, "metric": metric,
                   "delta_before": series[a], "delta_after": series[b],
                   "price": series[b] - series[a]}
            hit = price_ci.get((a, b, metric))
            if hit:
                row["price_ci"] = hit["price_ci"]
                row["paired"] = True
            else:
                row["paired"] = False
                row["note"] = ("no interval: this step changes which rows are scored, so the "
                               "two deltas are not paired and the difference is a difference "
                               "of point estimates only")
            guard_prices.append(row)
    guard_prices_by_key = {f"{r['from']}_{r['to']}_{r['metric']}": r for r in guard_prices}

    shipped = json.loads(Path(args.shipped).read_text()) if args.shipped else None
    repro = {"compared": [], "tolerance": args.repro_tol}
    if shipped:
        worst = 0.0
        for task, keys in SHIPPED_KEYS.items():
            for k in keys:
                got = by_id["L3"]["tasks"][task][k]
                want = shipped["tasks"][task][k]
                diff = abs(got - want)
                worst = max(worst, diff)
                repro["compared"].append({"task": task, "metric": k, "shipped": want,
                                          "ladder_L3": got, "abs_diff": diff})
        repro["max_abs_diff"] = worst
        repro["ok"] = bool(worst <= args.repro_tol)
        repro["note"] = ("L3 is the shipped protocol, so it must reproduce "
                         "results/backbone_transfer.json; the largest absolute difference over "
                         "the eight shipped point metrics is recorded either way.")
        print(f"[ladder] L3 vs shipped: max abs diff {worst:.3e} "
              f"(tolerance {args.repro_tol:.0e}) -> {'OK' if repro['ok'] else 'MISMATCH'}",
              flush=True)

    return {
        "seed": args.seed,
        "versions": versions_dict(),
        "generated_by": "scripts/fm/ladder_eval.py --check-able",
        "data_sources": meta["data_sources"],
        "labels": ["synthetic"],
        "what_this_is": (
            "One frozen checkpoint, no new pretraining. The same two downstream tasks are "
            "evaluated under four progressively stricter EVALUATION protocols, one guard at "
            "a time, to measure how much of a transfer lift is protocol rather than model."),
        "pretrain_guards_constant": {
            "label_excluded": True,
            "ids_hashed": True,
            "time_truncated": True,
            "note": ("These are pretrain-side guards and they stay ON at every rung. Only the "
                     "evaluation protocol varies."),
        },
        "held_fixed": {
            "checkpoint": "the full-corpus cardholder backbone (frozen, evaluation only)",
            "row_caps": {"train": args.max_train, "test": args.max_test},
            "merchant_embedding": ("pre-cut pooled at every rung; only the account-side "
                                   "embedding varies, so L0 is a lower bound on how much a "
                                   "fully permissive protocol inflates, not an upper bound"),
            "hyperparameters": "identical LightGBM settings and early stopping at every rung",
        },
        "bootstrap": {"method": "entity-clustered paired bootstrap", "B": args.bootstrap,
                      "ci": "percentile 95%"},
        "rungs": rungs,
        "rungs_by_id": by_id,
        "deltas_by_rung": deltas_by_rung,
        "guard_prices": guard_prices,
        "guard_prices_by_key": guard_prices_by_key,
        "l0_shared_accounts": n_shared,
        "l3_reproduces_shipped": repro,
        "check": {"command": "python scripts/fm/ladder_eval.py --stage run ... --check",
                  "tolerance": 1e-6,
                  "note": "recomputes every rung from the same frozen inputs and compares "
                          "every numeric leaf against the committed results/ladder.json"},
    }


# ---------------------------------------------------------------------- check ----

def numeric_leaves(obj, prefix=""):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield prefix, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from numeric_leaves(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from numeric_leaves(v, f"{prefix}/{i}")


SKIP_CHECK_PREFIXES = ("/versions",)


def compare(fresh: dict, stored: dict, tol: float) -> int:
    a = {k: v for k, v in numeric_leaves(fresh) if not k.startswith(SKIP_CHECK_PREFIXES)}
    b = {k: v for k, v in numeric_leaves(stored) if not k.startswith(SKIP_CHECK_PREFIXES)}
    bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append((k, a.get(k), b.get(k)))
        elif not math.isclose(a[k], b[k], rel_tol=0.0, abs_tol=tol):
            bad.append((k, a[k], b[k]))
    if bad:
        for k, x, y in bad[:25]:
            print(f"CHECK MISMATCH {k}: recomputed {x} vs stored {y}")
        print(f"CHECK FAILED: {len(bad)} numeric leaves differ beyond {tol:g}")
        return 5
    print(f"CHECK OK: {len(a)} numeric leaves reproduce within {tol:g}")
    return 0


# ----------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["user_emb", "run"])
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--eval", default="")
    ap.add_argument("--asof", default="")
    ap.add_argument("--user-emb", default="")
    ap.add_argument("--merchants", default="")
    ap.add_argument("--shipped", default="", help="backbone_transfer.json for the L3 assert")
    ap.add_argument("--max-train", type=int, default=800_000)
    ap.add_argument("--max-test", type=int, default=300_000)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--mcc-estimators", type=int, default=150)
    ap.add_argument("--fraud-estimators", type=int, default=600)
    ap.add_argument("--smoke", type=int, default=0,
                    help="code-path smoke: shrink every rung to N training rows; never reported")
    ap.add_argument("--repro-tol", type=float, default=1e-9)
    ap.add_argument("--check", action="store_true", help="recompute and compare at 1e-6; exit 0/5")
    ap.add_argument("--check-tol", type=float, default=1e-6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    seed_everything(args.seed)

    if args.stage == "user_emb":
        assert args.ckpt, "--ckpt required for the user_emb stage"
        stage_user_emb(args)
        return 0

    for req in ("eval", "asof", "user_emb", "merchants"):
        assert getattr(args, req), f"--{req.replace('_', '-')} required for the run stage"
    fresh = run_ladder(args)
    if args.check:
        stored = json.loads(Path(args.out).read_text())
        return compare(fresh, stored, args.check_tol)
    atomic_write_json(args.out, fresh)
    print(f"[ladder] wrote {args.out}", flush=True)
    repro = fresh.get("l3_reproduces_shipped", {})
    if repro.get("compared") and not repro.get("ok"):
        print("FAIL: L3 did not reproduce the shipped numbers within tolerance "
              f"(max abs diff {repro['max_abs_diff']:.3e}). Numbers are written as obtained.")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
