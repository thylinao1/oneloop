"""transfer_eval.py: multi-task transfer table under the leakage policy.

Tasks (post-cut transactions of HELD-OUT users; entity-disjoint split):
  fraud    : 'Is Fraud?'; AUC + PR-AUC
  next_mcc : predict the scored transaction's MCC from history only; top-1/top-5
Arms (identical rows + hyperparams):
  baseline : LightGBM on per-entity temporal aggregates (counts, amount
             mean/std/max, recency, frequency encodings) [+ current-txn fields
             for fraud only]
  with_emb : baseline + backbone as-of user embedding (fraud also gets the
             scored merchant's pre-cut embedding)
CIs: entity-clustered PAIRED bootstrap on the deltas.

Stages: --stage select  (writes eval_rows.npy for embed.py)
        --stage run     (consumes as-of + merchant embeddings, writes JSON)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from common import FIELDS, load_prep, seed_everything, atomic_write_json, versions_dict

TEST_USER_FRAC = 0.2
VAL_USER_FRAC = 0.1


def cap_rows(rows: np.ndarray, fraud: np.ndarray, cap: int, rng) -> np.ndarray:
    if cap <= 0 or len(rows) <= cap:
        return np.sort(rows)
    pos = rows[fraud[rows] == 1]
    neg = rows[fraud[rows] == 0]
    n_neg = max(0, cap - len(pos))
    keep = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
    return np.sort(np.concatenate([pos, keep]))


def stage_select(args):
    d = load_prep(args.prep)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    users = np.unique(d["user"])
    perm = rng.permutation(users)
    n_test = max(1, int(len(users) * TEST_USER_FRAC))
    test_users = set(perm[:n_test].tolist())
    post = d["ts"] >= d["meta"]["cut_ts"]
    in_test = np.isin(d["user"], list(test_users))
    test_rows = cap_rows(np.flatnonzero(post & in_test), d["fraud"], args.max_test, rng)
    train_rows = cap_rows(np.flatnonzero(post & ~in_test), d["fraud"], args.max_train, rng)
    np.savez(out / "select.npz", train_rows=train_rows, test_rows=test_rows,
             test_users=np.array(sorted(test_users)))
    eval_rows = np.concatenate([train_rows, test_rows])
    np.save(out / "eval_rows.npy", eval_rows)
    meta = {"n_users": len(users), "n_test_users": n_test, "n_train_rows": int(len(train_rows)),
            "n_test_rows": int(len(test_rows)), "seed": args.seed,
            "row_sampling": "uniform + all fraud positives kept",
            "split": "entity-disjoint by user, pre-registered at select stage"}
    atomic_write_json(out / "select_meta.json", meta)
    print(f"[select] {meta}", flush=True)


def history_features(d) -> pl.DataFrame:
    """Per-row strictly-before aggregates (vectorized over all rows)."""
    fi = {f: i for i, f in enumerate(FIELDS)}
    df = pl.DataFrame({
        "user": d["user"], "ts": d["ts"], "amount": d["amount"].astype(np.float64),
        "mcc_tok": d["tokens"][:, fi["mcc"]].astype(np.int32),
        "city_tok": d["tokens"][:, fi["city"]].astype(np.int32),
        "state_tok": d["tokens"][:, fi["state"]].astype(np.int32),
        "hour_tok": d["tokens"][:, fi["hour"]].astype(np.int32),
        "chip_tok": d["tokens"][:, fi["use_chip"]].astype(np.int32),
    })
    df = df.with_columns(
        (pl.col("amount").cum_count().over("user") - 1).alias("hist_count"),
        pl.col("amount").cum_sum().shift(1).over("user").alias("_s1"),
        (pl.col("amount") ** 2).cum_sum().shift(1).over("user").alias("_s2"),
        pl.col("amount").cum_max().shift(1).over("user").alias("amt_max"),
        pl.col("amount").shift(1).over("user").alias("last_amount"),
        (pl.col("ts") - pl.col("ts").shift(1).over("user")).alias("recency_s"),
        (pl.col("ts") - pl.col("ts").first().over("user")).alias("since_first_s"),
        pl.col("mcc_tok").shift(1).over("user").alias("prev_mcc"),
        pl.col("city_tok").shift(1).over("user").alias("prev_city"),
    ).with_columns(
        (pl.col("_s1") / pl.col("hist_count")).alias("amt_mean"),
    ).with_columns(
        (pl.col("_s2") / pl.col("hist_count") - pl.col("amt_mean") ** 2)
        .clip(lower_bound=0).sqrt().alias("amt_std"),
    )
    return df


def freq_encode(tok: np.ndarray, counts: np.ndarray) -> np.ndarray:
    tok = np.clip(tok, 0, len(counts) - 1)
    return counts[tok]


def build_matrices(d, hf: pl.DataFrame, rows, asof, merch_emb, merch_has, freq, task: str):
    """Return (X_base, X_emb) float32 for given rows (rows must be sorted ascending)."""
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
        X_base = np.hstack([hist, cur])
        me = merch_emb[d["merchant"][rows]]
        mh = merch_has[d["merchant"][rows]][:, None].astype(np.float32)
        X_emb = np.hstack([X_base, asof, me, mh])
    else:  # next_mcc: history-only features (the scored txn is the answer)
        X_base = hist
        X_emb = np.hstack([X_base, asof])
    return X_base, X_emb


def paired_cluster_bootstrap(user_of_row, delta_fns, B, seed):
    uniq, inv = np.unique(user_of_row, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    bnd = np.searchsorted(sorted_inv, np.arange(len(uniq)))
    bnd = np.append(bnd, len(sorted_inv))
    rows_by_user = [order[bnd[i]:bnd[i + 1]] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.full((B, len(delta_fns)), np.nan)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_user[p] for p in pick])
        for k, fn in enumerate(delta_fns):
            try:
                out[b, k] = fn(idx)
            except ValueError:
                pass  # degenerate resample (single class) -> nan
    return [
        [float(np.nanpercentile(out[:, k], 2.5)), float(np.nanpercentile(out[:, k], 97.5))]
        for k in range(len(delta_fns))
    ]


def stage_run(args):
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    seed_everything(args.seed)
    d = load_prep(args.prep)
    meta = d["meta"]
    ev = Path(args.eval)
    sel = np.load(ev / "select.npz")
    train_rows, test_rows = sel["train_rows"], sel["test_rows"]
    eval_rows = np.load(ev / "eval_rows.npy")
    asof_all = np.load(args.asof)
    assert len(asof_all) == len(eval_rows), "as-of embeddings misaligned with eval rows"
    pos_of = {int(r): i for i, r in enumerate(eval_rows)}
    asof_train = asof_all[[pos_of[int(r)] for r in train_rows]]
    asof_test = asof_all[[pos_of[int(r)] for r in test_rows]]

    mdf = pl.read_parquet(args.merchants)
    emb_cols = [c for c in mdf.columns if c.startswith("emb_")]
    emb_dim = len(emb_cols)
    n_merch = int(d["merchant"].max()) + 1
    merch_emb = np.zeros((n_merch, emb_dim), dtype=np.float32)
    merch_has = np.zeros(n_merch, dtype=bool)
    mid = mdf["merchant_id"].to_numpy()
    merch_emb[mid] = mdf.select(emb_cols).to_numpy().astype(np.float32)
    merch_has[mid] = True

    # frequency encodings from PRE-cut rows of TRAIN users only (no test-entity info)
    pre = d["ts"] < meta["cut_ts"]
    test_users = set(sel["test_users"].tolist())
    train_pre = pre & ~np.isin(d["user"], list(test_users))
    fi = {f: i for i, f in enumerate(FIELDS)}
    freq = {}
    for name in ("mcc", "city", "state"):
        tok = d["tokens"][:, fi[name]].astype(np.int64)
        cnt = np.bincount(tok[train_pre], minlength=meta["vocab_sizes"][fi[name]]).astype(np.float64)
        freq[name] = (cnt / max(1, cnt.sum())).astype(np.float32)

    print("[run] building history features...", flush=True)
    hf = history_features(d)

    summary = json.loads(Path(args.ckpt_summary).read_text())
    results_tasks = {}
    boot_meta = {"method": "entity-clustered paired bootstrap", "B": args.bootstrap,
                 "ci": "percentile 95%"}

    rng = np.random.default_rng(args.seed)
    tr_users = d["user"][train_rows]
    uniq_tr = np.unique(tr_users)
    val_users = set(rng.permutation(uniq_tr)[: max(1, int(len(uniq_tr) * VAL_USER_FRAC))].tolist())
    val_mask = np.isin(tr_users, list(val_users))

    for task in ("fraud", "next_mcc"):
        print(f"[run] === task {task} ===", flush=True)
        Xb_tr, Xe_tr = build_matrices(d, hf, train_rows, asof_train, merch_emb, merch_has, freq, task)
        Xb_te, Xe_te = build_matrices(d, hf, test_rows, asof_test, merch_emb, merch_has, freq, task)
        if task == "fraud":
            y_tr, y_te = d["fraud"][train_rows], d["fraud"][test_rows]
            params = dict(objective="binary", metric="average_precision", learning_rate=0.05,
                          num_leaves=63, n_estimators=600, seed=args.seed,
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
            print(f"[run] {task}/{arm}: trained {clf.best_iteration_ or params['n_estimators']} iters, "
                  f"{Xtr.shape[1]} features", flush=True)
        u_te = d["user"][test_rows]
        if task == "fraud":
            s_b, s_e = preds["baseline"][:, 1], preds["with_emb"][:, 1]
            res = {
                "baseline_auc": float(roc_auc_score(y_te, s_b)),
                "baseline_prauc": float(average_precision_score(y_te, s_b)),
                "with_emb_auc": float(roc_auc_score(y_te, s_e)),
                "with_emb_prauc": float(average_precision_score(y_te, s_e)),
            }
            fns = [
                lambda i: roc_auc_score(y_te[i], s_e[i]) - roc_auc_score(y_te[i], s_b[i]),
                lambda i: average_precision_score(y_te[i], s_e[i]) - average_precision_score(y_te[i], s_b[i]),
            ]
            cis = paired_cluster_bootstrap(u_te, fns, args.bootstrap, args.seed)
            res["delta_auc_ci"], res["delta_prauc_ci"] = cis
            res["n_test_pos"] = int(y_te.sum())
        else:
            def topk(p, y, k):
                top = np.argsort(-p, axis=1)[:, :k]
                return (top == y[:, None]).any(axis=1).astype(np.float64)
            hit1_b, hit5_b = topk(preds["baseline"], y_te, 1), topk(preds["baseline"], y_te, 5)
            hit1_e, hit5_e = topk(preds["with_emb"], y_te, 1), topk(preds["with_emb"], y_te, 5)
            res = {
                "baseline_top1": float(hit1_b.mean()), "baseline_top5": float(hit5_b.mean()),
                "with_emb_top1": float(hit1_e.mean()), "with_emb_top5": float(hit5_e.mean()),
                "n_classes": meta["n_mcc_classes"],
                "features": "history-only (scored txn fields excluded)",
            }
            fns = [
                lambda i: hit1_e[i].mean() - hit1_b[i].mean(),
                lambda i: hit5_e[i].mean() - hit5_b[i].mean(),
            ]
            cis = paired_cluster_bootstrap(u_te, fns, args.bootstrap, args.seed)
            res["delta_top1_ci"], res["delta_top5_ci"] = cis
        res["n_train"], res["n_test"] = int(len(train_rows)), int(len(test_rows))
        res["n_test_users"] = int(len(np.unique(u_te)))
        results_tasks[task] = res
        print(f"[run] {task}: {res}", flush=True)

    out = {
        "seed": args.seed,
        "versions": versions_dict(),
        "generated_by": "scripts/fm/transfer_eval.py --check-able",
        "data_sources": meta["data_sources"],
        "labels": ["synthetic"],
        "pretrain": {
            "params_m": summary["params_m"],
            "epochs": summary["epochs"],
            "loss_curve": summary["loss_curve"],
            "corpus_cut_date": summary["corpus_cut_date"],
        },
        "tasks": results_tasks,
        "leakage_checks": {
            "label_excluded": bool(meta["leakage"]["label_excluded"]),
            "time_truncated": bool(summary["time_truncated"]),
            "as_of_embeddings": True,  # embed.py: strictly-before windows only
            "ids_hashed": bool(meta["leakage"]["ids_excluded"]),  # excluded entirely
        },
        "split": json.loads((ev / "select_meta.json").read_text()),
        "bootstrap": boot_meta,
    }
    atomic_write_json(args.out, out)
    print(f"[run] wrote {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "run"])
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True, help="select: dir | run: output json path")
    ap.add_argument("--eval", default="", help="run: dir from select stage")
    ap.add_argument("--asof", default="")
    ap.add_argument("--merchants", default="")
    ap.add_argument("--ckpt-summary", default="")
    ap.add_argument("--max-train", type=int, default=800_000)
    ap.add_argument("--max-test", type=int, default=300_000)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--mcc-estimators", type=int, default=150)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    seed_everything(args.seed)
    if args.stage == "select":
        stage_select(args)
    else:
        for req in ("eval", "asof", "merchants", "ckpt_summary"):
            assert getattr(args, req), f"--{req.replace('_','-')} required for run stage"
        stage_run(args)


if __name__ == "__main__":
    main()
