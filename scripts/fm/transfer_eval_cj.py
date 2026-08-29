"""transfer_eval_cj.py - L3 transfer evaluation on the Complete Journey corpus.

Why this file exists instead of reusing transfer_eval.py unchanged: transfer_eval.py
hard-codes the TabFormer surface in three places no flag can bridge (the two-task loop
over ("fraud", "next_mcc"), the field names "mcc"/"city"/"state"/"hour"/"use_chip"
inside history_features and the frequency encodings, and `from common import FIELDS`
bound at import time). The bootstrap machinery and the protocol constants ARE shared:
paired_cluster_bootstrap and VAL_USER_FRAC are imported from transfer_eval.py, so the
interval construction is the same code object the shipped numbers used.

Task: next COMMODITY_DESC (top-30 + other), the analog of the shipped next-MCC task.
Protocol: L3 ONLY, exactly the shipped rungs' strictest form:
  - household-disjoint split (the entity here is the household, ~2,500 of them;
    split sizes are recorded in the output because the entity pool is small)
  - as-of prefix-only embeddings (strictly before the scored transaction; embed.py)
  - baseline equipped with per-entity temporal aggregates (the strengthened baseline)
  - identical LightGBM hyperparameters in both arms, entity-clustered paired bootstrap

THE SCORED UNIT IS THE BASKET-OPENING TRANSACTION. Complete Journey rows arrive in
baskets that share one timestamp (about 9-10 items per basket against embed.py's window
of 16). A row scored mid-basket would receive an as-of window filled mostly with its OWN
basket's other items, whose commodity tokens are in the vocabulary, so the delta would
measure within-basket co-occurrence, not transfer. The select stage therefore restricts
scored rows to each post-cut basket's FIRST row, additionally requiring the preceding
row to carry a strictly smaller ts (two baskets of one household can share a timestamp),
so every as-of window contains only strictly-earlier-ts rows while embed.py stays byte
identical. The baseline's history features are aggregated over strictly-earlier ts
(prior baskets) for BOTH arms; see history_features_cj.

Two pretraining seeds (7 and 8) are evaluated against the SAME select split with the
run-stage evaluation seed PINNED (job scripts pass --seed 42 to both runs), so the
validation carve-out, the LightGBM seed and the bootstrap draws are identical across
backbone seeds and seed-to-seed differences reflect pretraining variability only.

Leakage hard-fails inherited/enforced HERE:
  - rows scored are post-cut rows of held-out households only (asserted disjoint)
  - every scored row opens a basket and its predecessor has strictly smaller ts
    (asserted at select AND re-verified independently at run)
  - frequency encodings fit on pre-cut rows of train households only
  - the scored transaction's own fields never enter the features (history-only), and
    no same-basket item enters them either (strictly-earlier-ts aggregation)

Additions over the shipped script, labeled as additions: majority_floor_top1/top5
(predict the most frequent pre-registered classes for every row). transfer_eval.py does
not report this floor; it is reported here because next-commodity class balance on a
grocery corpus is sharper than MCC on TabFormer and a lift over a strong floor is the
honest denominator.

Stages: --stage select   (writes select.npz + eval_rows.npy for embed.py)
        --stage run      (per backbone seed -> one JSON)
        --stage combine  (two per-seed JSONs -> results file with required_sentence)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm  # noqa: F401  MUST load before torch: seed_everything imports torch,
                 # and on macOS the two OpenMP runtimes coexist only if LightGBM's
                 # loads first (segfault otherwise; harmless on the cluster's Linux)
import numpy as np
import polars as pl

from common import atomic_write_json, seed_everything, versions_dict
from cj_common import CJ_FIELDS, load_prep_cj
from transfer_eval import VAL_USER_FRAC, paired_cluster_bootstrap

TEST_USER_FRAC = 0.2


def cap_uniform(rows: np.ndarray, cap: int, rng) -> np.ndarray:
    """Uniform row cap (no label stratification: the scored task is next commodity)."""
    if cap <= 0 or len(rows) <= cap:
        return np.sort(rows)
    return np.sort(rng.choice(rows, size=cap, replace=False))


def scored_row_mask(user: np.ndarray, ts: np.ndarray, basket: np.ndarray):
    """Masks over the (household, ts)-sorted rows.

    first[i]:  row i opens a basket (segment start or basket id change)
    strict[i]: row i's predecessor within the household has strictly smaller ts
               (or i starts the household's segment)
    A row is scoreable iff first & strict: its as-of window (embed.py fills it with the
    rows immediately before i) then contains only strictly-earlier-ts transactions.
    first & ~strict happens when two baskets of one household share a timestamp; those
    basket-opening rows are excluded and counted.
    """
    n = len(user)
    first = np.ones(n, dtype=bool)
    first[1:] = (user[1:] != user[:-1]) | (basket[1:] != basket[:-1])
    strict = np.ones(n, dtype=bool)
    strict[1:] = (user[1:] != user[:-1]) | (ts[1:] > ts[:-1])
    return first, strict


def assert_scored_rows_clean(rows: np.ndarray, user: np.ndarray, ts: np.ndarray,
                             basket: np.ndarray, where: str) -> None:
    """Independent recheck (not the construction path): every scored row opens its
    basket and no same-ts row precedes it within the household."""
    r = rows[rows > 0]
    same_user = user[r - 1] == user[r]
    assert (ts[r - 1][same_user] < ts[r][same_user]).all(), \
        f"LEAKAGE ({where}): a scored row has a same-ts predecessor in its window"
    assert (basket[r - 1][same_user] != basket[r][same_user]).all(), \
        f"LEAKAGE ({where}): a scored row does not open its basket"


# ------------------------------------------------------------------- select ----

def stage_select(args):
    d = load_prep_cj(args.prep)
    basket = np.load(Path(args.prep) / "basket.npy")
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    users = np.unique(d["user"])
    perm = rng.permutation(users)
    n_test = max(1, int(len(users) * TEST_USER_FRAC))
    test_users = set(perm[:n_test].tolist())
    post = d["ts"] >= d["meta"]["cut_ts"]
    in_test = np.isin(d["user"], list(test_users))

    # scored unit = the basket-opening transaction (see module docstring): rows scored
    # mid-basket would hand the as-of window their own basket's items
    first, strict = scored_row_mask(d["user"], d["ts"], basket)
    scored_ok = first & strict
    n_post = int(post.sum())
    n_post_basket_start = int((post & first).sum())
    n_dropped_same_ts_tie = int((post & first & ~strict).sum())

    test_rows = cap_uniform(np.flatnonzero(post & in_test & scored_ok), args.max_test, rng)
    train_rows = cap_uniform(np.flatnonzero(post & ~in_test & scored_ok), args.max_train, rng)
    for rows, where in ((train_rows, "select/train"), (test_rows, "select/test")):
        assert scored_ok[rows].all(), f"scored-row mask violated at {where}"
        assert_scored_rows_clean(rows, d["user"], d["ts"], basket, where)
    print(f"[select] post-cut rows {n_post} -> basket-opening {n_post_basket_start} "
          f"(dropped {n_dropped_same_ts_tie} same-ts-tie basket starts)", flush=True)

    np.savez(out / "select.npz", train_rows=train_rows, test_rows=test_rows,
             test_users=np.array(sorted(test_users)))
    eval_rows = np.concatenate([train_rows, test_rows])
    np.save(out / "eval_rows.npy", eval_rows)
    meta = {"n_users": int(len(users)), "n_test_users": int(n_test),
            "n_train_users": int(len(users) - n_test),
            "n_train_rows": int(len(train_rows)), "n_test_rows": int(len(test_rows)),
            "seed": args.seed,
            "scored_unit": "basket-opening transaction (first row of each post-cut "
                           "basket, predecessor strictly earlier in ts)",
            "n_post_rows": n_post,
            "n_post_basket_opening_rows": n_post_basket_start,
            "n_basket_starts_dropped_same_ts_tie": n_dropped_same_ts_tie,
            "row_sampling": "uniform over basket-opening rows (no label stratification)",
            "split": "entity-disjoint by household, pre-registered at select stage",
            "small_entity_pool_note": "~2,500 households total; the split sizes above "
                                      "are the whole story, there is no larger pool"}
    atomic_write_json(out / "select_meta.json", meta)
    print(f"[select] {meta}", flush=True)


# ----------------------------------------------------------------- features ----

def history_features_cj(d) -> pl.DataFrame:
    """Per-row aggregates over STRICTLY-EARLIER-ts rows (prior baskets).

    transfer_eval.history_features uses row-level shift(1)/cum_* because TabFormer rows
    carry distinct timestamps. Complete Journey rows arrive in baskets sharing one
    timestamp, so a row-level shift would let same-basket items into the history. Here
    every aggregate is computed over (user, ts) time groups and shifted by one GROUP:
    all rows of a basket carry identical features built only from strictly earlier ts.
    Same ten columns, same names, consumed identically by BOTH arms. prev_com/prev_dep/
    last_amount come from the latest row of the previous time group.
    """
    fi = {f: i for i, f in enumerate(CJ_FIELDS)}
    df = pl.DataFrame({
        "user": d["user"], "ts": d["ts"], "amount": d["amount"].astype(np.float64),
        "com_tok": d["tokens"][:, fi["commodity"]].astype(np.int32),
        "dep_tok": d["tokens"][:, fi["department"]].astype(np.int32),
    }).with_row_index("_ri")
    grp = (
        df.group_by(["user", "ts"], maintain_order=True)
        .agg(
            pl.len().alias("_n"),
            pl.col("amount").sum().alias("_a1"),
            (pl.col("amount") ** 2).sum().alias("_a2"),
            pl.col("amount").max().alias("_amax"),
            pl.col("amount").last().alias("_alast"),
            pl.col("com_tok").last().alias("_com_last"),
            pl.col("dep_tok").last().alias("_dep_last"),
        )
        .with_columns(
            pl.col("_n").cum_sum().shift(1).over("user").alias("hist_count"),
            pl.col("_a1").cum_sum().shift(1).over("user").alias("_s1"),
            pl.col("_a2").cum_sum().shift(1).over("user").alias("_s2"),
            pl.col("_amax").cum_max().shift(1).over("user").alias("amt_max"),
            pl.col("_alast").shift(1).over("user").alias("last_amount"),
            pl.col("_com_last").shift(1).over("user").alias("prev_com"),
            pl.col("_dep_last").shift(1).over("user").alias("prev_dep"),
            (pl.col("ts") - pl.col("ts").shift(1).over("user")).alias("recency_s"),
            (pl.col("ts") - pl.col("ts").first().over("user")).alias("since_first_s"),
        )
        .with_columns((pl.col("_s1") / pl.col("hist_count")).alias("amt_mean"))
        .with_columns(
            (pl.col("_s2") / pl.col("hist_count") - pl.col("amt_mean") ** 2)
            .clip(lower_bound=0).sqrt().alias("amt_std"),
        )
    )
    out = (
        df.join(
            grp.select("user", "ts", "hist_count", "amt_mean", "amt_std", "amt_max",
                       "last_amount", "recency_s", "since_first_s", "prev_com", "prev_dep"),
            on=["user", "ts"], how="left",
        )
        .sort("_ri")
        .drop("_ri")
    )
    assert out.height == df.height, "history feature join changed the row count"
    return out


def freq_encode(tok: np.ndarray, counts: np.ndarray) -> np.ndarray:
    tok = np.clip(tok, 0, len(counts) - 1)
    return counts[tok]


def build_matrices(d, hf: pl.DataFrame, rows, asof, freq):
    """(X_base, X_emb) for `rows` (sorted ascending). History-only: the scored
    transaction's commodity is the answer, so none of its own fields are inputs."""
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
        freq_encode(sub["prev_com"].fill_null(0).to_numpy(), freq["commodity"]),
        freq_encode(sub["prev_dep"].fill_null(0).to_numpy(), freq["department"]),
        sub["prev_com"].fill_null(0).to_numpy(),
    ]).astype(np.float32)
    X_base = hist
    X_emb = np.hstack([X_base, asof])
    return X_base, X_emb


# --------------------------------------------------------------------- run ----

def stage_run(args):
    import lightgbm as lgb

    seed_everything(args.seed)
    d = load_prep_cj(args.prep)
    meta = d["meta"]
    ev = Path(args.eval)
    sel = np.load(ev / "select.npz")
    train_rows, test_rows = sel["train_rows"], sel["test_rows"]
    test_users = set(sel["test_users"].tolist())

    # hard-fail leakage guards
    ov = np.intersect1d(np.unique(d["user"][train_rows]), np.unique(d["user"][test_rows]))
    assert len(ov) == 0, f"LEAKAGE: {len(ov)} households shared across the split"
    post = d["ts"] >= meta["cut_ts"]
    assert post[train_rows].all() and post[test_rows].all(), \
        "LEAKAGE: pre-cut rows in the scored pool"
    # basket guard, re-verified here independently of the select stage: every scored row
    # opens its basket and no same-ts row precedes it (so embed.py's as-of window holds
    # only strictly-earlier-ts transactions)
    basket = np.load(Path(args.prep) / "basket.npy")
    assert_scored_rows_clean(train_rows, d["user"], d["ts"], basket, "run/train")
    assert_scored_rows_clean(test_rows, d["user"], d["ts"], basket, "run/test")

    eval_rows = np.load(ev / "eval_rows.npy")
    asof_all = np.load(args.asof)
    assert len(asof_all) == len(eval_rows), "as-of embeddings misaligned with eval rows"
    pos_of = {int(r): i for i, r in enumerate(eval_rows)}
    asof_train = asof_all[[pos_of[int(r)] for r in train_rows]]
    asof_test = asof_all[[pos_of[int(r)] for r in test_rows]]

    # frequency encodings from PRE-cut rows of TRAIN households only (no test-entity info)
    pre = d["ts"] < meta["cut_ts"]
    train_pre = pre & ~np.isin(d["user"], list(test_users))
    fi = {f: i for i, f in enumerate(CJ_FIELDS)}
    freq = {}
    for name in ("commodity", "department"):
        tok = d["tokens"][:, fi[name]].astype(np.int64)
        cnt = np.bincount(tok[train_pre], minlength=meta["vocab_sizes"][fi[name]]).astype(np.float64)
        freq[name] = (cnt / max(1, cnt.sum())).astype(np.float32)

    print("[run] building history features...", flush=True)
    hf = history_features_cj(d)

    summary = json.loads(Path(args.ckpt_summary).read_text())

    rng = np.random.default_rng(args.seed)
    tr_users = d["user"][train_rows]
    uniq_tr = np.unique(tr_users)
    val_users = set(rng.permutation(uniq_tr)[: max(1, int(len(uniq_tr) * VAL_USER_FRAC))].tolist())
    val_mask = np.isin(tr_users, list(val_users))

    Xb_tr, Xe_tr = build_matrices(d, hf, train_rows, asof_train, freq)
    Xb_te, Xe_te = build_matrices(d, hf, test_rows, asof_test, freq)
    y_tr, y_te = d["mcc_class"][train_rows], d["mcc_class"][test_rows]
    # identical hyperparameters to transfer_eval.py's next_mcc task
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
        print(f"[run] next_commodity/{arm}: trained "
              f"{clf.best_iteration_ or params['n_estimators']} iters, "
              f"{Xtr.shape[1]} features", flush=True)

    def topk(p, y, k):
        top = np.argsort(-p, axis=1)[:, :k]
        return (top == y[:, None]).any(axis=1).astype(np.float64)

    hit1_b, hit5_b = topk(preds["baseline"], y_te, 1), topk(preds["baseline"], y_te, 5)
    hit1_e, hit5_e = topk(preds["with_emb"], y_te, 1), topk(preds["with_emb"], y_te, 5)
    u_te = d["user"][test_rows]

    # majority floor (addition over the shipped script, labeled as such): predict the
    # k most frequent TRAIN classes for every test row
    cls_freq = np.bincount(y_tr, minlength=meta["n_mcc_classes"])
    order = np.argsort(-cls_freq)
    floor1 = float((y_te == order[0]).mean())
    floor5 = float(np.isin(y_te, order[:5]).mean())

    res = {
        "baseline_top1": float(hit1_b.mean()), "baseline_top5": float(hit5_b.mean()),
        "with_emb_top1": float(hit1_e.mean()), "with_emb_top5": float(hit5_e.mean()),
        "n_classes": meta["n_mcc_classes"],
        "features": "history-only, aggregated over strictly-earlier ts (prior baskets); "
                    "the scored unit is the basket-opening transaction",
        "majority_floor_top1": floor1,
        "majority_floor_top5": floor5,
        "majority_floor_note": "predict the most frequent train classes for every row; "
                               "not reported by the shipped transfer_eval.py, added here "
                               "because grocery commodity balance is sharp",
    }
    res["delta_top1"] = res["with_emb_top1"] - res["baseline_top1"]
    res["delta_top5"] = res["with_emb_top5"] - res["baseline_top5"]
    fns = [
        lambda i: hit1_e[i].mean() - hit1_b[i].mean(),
        lambda i: hit5_e[i].mean() - hit5_b[i].mean(),
    ]
    cis = paired_cluster_bootstrap(u_te, fns, args.bootstrap, args.seed)
    res["delta_top1_ci"], res["delta_top5_ci"] = cis
    res["n_train"], res["n_test"] = int(len(train_rows)), int(len(test_rows))
    res["n_test_users"] = int(len(np.unique(u_te)))
    print(f"[run] next_commodity: {res}", flush=True)

    out = {
        "seed": args.seed,
        "versions": versions_dict(),
        "generated_by": "scripts/fm/transfer_eval_cj.py",
        "data_sources": meta["data_sources"],
        "labels": ["real"],
        "protocol": "L3 (household-disjoint split, as-of prefix-only embeddings, "
                    "aggregate-equipped baseline)",
        "pretrain": {
            "params_m": summary["params_m"],
            "epochs": summary["epochs"],
            "loss_curve": summary["loss_curve"],
            "corpus_cut_date": summary["corpus_cut_date"],
            "backbone_seed": summary["seed"],
        },
        "tasks": {"next_commodity": res},
        "leakage_checks": {
            "label_excluded": bool(meta["leakage"]["label_excluded"]),
            "time_truncated": bool(summary["time_truncated"]),
            "as_of_embeddings": True,  # embed.py windows; scored rows open baskets, so
                                       # the window holds strictly-earlier-ts rows only
            "scored_unit_basket_opening": True,  # asserted above, both split sides
            "ids_excluded": bool(meta["leakage"]["ids_excluded"]),
        },
        "split": json.loads((ev / "select_meta.json").read_text()),
        "bootstrap": {"method": "entity-clustered paired bootstrap (household clusters); "
                                "same code object as the shipped transfer_eval.py",
                      "B": args.bootstrap, "ci": "percentile 95%"},
    }
    atomic_write_json(args.out, out)
    print(f"[run] wrote {args.out}", flush=True)


# ------------------------------------------------------------------ combine ----

def _seed_block(o: dict) -> dict:
    t = o["tasks"]["next_commodity"]
    return {
        "backbone_seed": o["pretrain"]["backbone_seed"],
        "eval_seed": o["seed"],
        "baseline_top1": t["baseline_top1"], "with_emb_top1": t["with_emb_top1"],
        "baseline_top5": t["baseline_top5"], "with_emb_top5": t["with_emb_top5"],
        "delta_top1": t["delta_top1"], "delta_top1_ci": t["delta_top1_ci"],
        "delta_top5": t["delta_top5"], "delta_top5_ci": t["delta_top5_ci"],
        "majority_floor_top1": t["majority_floor_top1"],
        "majority_floor_top5": t["majority_floor_top5"],
    }


def stage_combine(args):
    inputs = [json.loads(Path(p).read_text()) for p in args.inputs]
    assert len(inputs) == 2, "combine expects exactly the two pre-registered seed files"
    blocks = [_seed_block(o) for o in inputs]
    seeds = [b["backbone_seed"] for b in blocks]
    assert len(set(seeds)) == 2, "the two inputs must come from different backbone seeds"
    assert blocks[0]["eval_seed"] == blocks[1]["eval_seed"], (
        "eval seeds differ between the two runs; the run stage must be invoked with the "
        "same pinned --seed for both backbones so seed-to-seed differences reflect "
        "pretraining variability only")

    los = [b["delta_top1_ci"][0] for b in blocks]
    his = [b["delta_top1_ci"][1] for b in blocks]
    d1 = [b["delta_top1"] for b in blocks]
    if all(lo > 0 for lo in los):
        sentence = (
            "Under the L3 protocol on the real Complete Journey corpus the backbone's "
            f"transfer gain on next-commodity top-1 is positive in both pretraining seeds "
            f"({d1[0]:+.4f} and {d1[1]:+.4f}), with both 95% household-clustered intervals "
            "above zero.")
        call = "positive_both_seeds"
    elif all(hi < 0 for hi in his):
        sentence = (
            "Under the L3 protocol on the real Complete Journey corpus the backbone's "
            f"transfer gain on next-commodity top-1 is negative in both pretraining seeds "
            f"({d1[0]:+.4f} and {d1[1]:+.4f}), with both 95% household-clustered intervals "
            "below zero. Per the pre-registration this ships as obtained.")
        call = "negative_both_seeds"
    else:
        sentence = (
            "Under the L3 protocol on the real Complete Journey corpus the backbone's "
            f"transfer gain on next-commodity top-1 ({d1[0]:+.4f} and {d1[1]:+.4f} across "
            "the two pretraining seeds) does not separate from zero in at least one seed's "
            "95% household-clustered interval. Per the pre-registration this ships as a "
            "null or mixed result, not as a positive.")
        call = "null_or_mixed"

    out = {
        "generated_by": "scripts/fm/transfer_eval_cj.py --stage combine",
        "versions": versions_dict(),
        "seeds": seeds,
        "eval_seed": blocks[0]["eval_seed"],
        "data_sources": inputs[0]["data_sources"],
        "labels": ["real"],
        "what_this_is": "the shipped L3 transfer protocol replicated on a real retail "
                        "corpus: same guards, same bootstrap code, two pretraining seeds "
                        "against one pre-registered household-disjoint split",
        "scale_caveat": "2.6M transactions and ~2,500 households against TabFormer's 24M "
                        "rows: a protocol replication on a small real corpus, not a scale "
                        "replication",
        "per_seed": {str(b["backbone_seed"]): b for b in blocks},
        "split": inputs[0]["split"],
        "leakage_checks": inputs[0]["leakage_checks"],
        "bootstrap": inputs[0]["bootstrap"],
        "decision_rule": "pre-registered in CJ-REPLICATION-PREREG.md: positive only if "
                         "both seed intervals sit above zero; anything else ships as a "
                         "null or mixed result; no third seed, no protocol changes after "
                         "seeing numbers",
        "required_sentence": sentence,
        "call": call,
    }
    atomic_write_json(args.out, out)
    print(f"[combine] call={call}")
    print(f"[combine] {sentence}")
    print(f"[combine] wrote {args.out}", flush=True)


# --------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "run", "combine"])
    ap.add_argument("--prep", default="")
    ap.add_argument("--out", required=True, help="select: dir | run/combine: output json path")
    ap.add_argument("--eval", default="", help="run: dir from select stage")
    ap.add_argument("--asof", default="")
    ap.add_argument("--ckpt-summary", default="")
    ap.add_argument("--inputs", nargs="*", default=[], help="combine: two per-seed JSONs")
    ap.add_argument("--max-train", type=int, default=800_000,
                    help="cap only; the CJ post-cut pool is smaller, so it rarely binds")
    ap.add_argument("--max-test", type=int, default=300_000)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--mcc-estimators", type=int, default=150)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7,
                    help="select: split seed. run: EVAL seed (val carve-out, LightGBM, "
                         "bootstrap); pin it to one constant across backbone seeds - the "
                         "backbone seed enters only through --asof/--ckpt-summary paths")
    args = ap.parse_args()
    seed_everything(args.seed)
    if args.stage == "select":
        assert args.prep, "--prep required for select"
        stage_select(args)
    elif args.stage == "run":
        for req in ("prep", "eval", "asof", "ckpt_summary"):
            assert getattr(args, req), f"--{req.replace('_', '-')} required for run stage"
        stage_run(args)
    else:
        assert args.inputs, "--inputs required for combine"
        stage_combine(args)


if __name__ == "__main__":
    main()
