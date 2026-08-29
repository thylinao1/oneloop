"""protection_pll.py: masked-field pseudo-log-likelihood as a LABEL-FREE surprise score.

WHAT THIS IS
  The frozen backbone was pretrained with a masked-field objective at mask
  probability 0.15 (pretrain.py). That makes masked-field pseudo-log-likelihood a
  direct readout with no retraining and no fine-tuning: for one scored
  transaction, mask ONE field at a time, run the encoder over that account's
  window ending at the scored transaction, and read the negative log-likelihood
  of the token that was actually observed. Summing those ten per-field terms
  gives the transaction's behavioural surprise given the account's own history.
  PLL as an evaluation readout for masked language models is Salazar et al.,
  ACL 2020 ("Masked Language Model Scoring").

  NOTE ON VARIANTS: masking every field of the scored position at once is OUT OF
  DISTRIBUTION for a 15% mask-probability pretrain (the model never saw a fully
  blank row during training). We do not ship a full-mask variant. If one is ever
  added it must be labelled separately in the results file.

ZERO FRAUD LABELS ENTER THE SCORE
  Enforced structurally, not by assertion in prose:
    * build_scores() is handed tokens, row indices, segment starts, the frozen
      checkpoint and marginal token counts. The fraud array is not in its scope.
    * main() computes every score FIRST, records the sha256 of the score matrix,
      and only then opens prep/fraud.npy. The hash is re-checked at reporting
      time, so the score bytes provably predate any label in memory.
  Labels are used afterwards and only to measure how well the ranking separates.

THE HONEST SKEPTIC'S CONTROL
  A surprise score can be nothing more than "this field value is globally rare".
  So we build a UNIGRAM RARITY control from marginal token frequencies with no
  model at all, fit on pre-cut rows of non-test accounts, and score it exactly
  the same way. Both ship whichever way the comparison lands.

  A third variant, an unweighted z-score sum of the two, is also reported. It
  uses no labels either (standardization is over the scored rows).

ERRORS-EXCLUDED VARIANT
  TabFormer carries an 'Errors?' field which may co-occur with fraud. Every
  score is therefore reported twice, once over all ten fields and once over the
  nine fields with 'errors' dropped, so nobody can say the score is reading an
  error flag.

Stages: --stage run          score + measure -> protection.json
        --stage run --check  recompute and compare every numeric leaf at 1e-6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from common import (
    FIELDS, MASK_ID, PAD_ID, N_SPECIAL, atomic_write_json, seed_everything,
    user_segments, versions_dict,
)

ERRORS_FIELD = "errors"
TOPK_FRACS = (0.001, 0.01)

# Field sets. "all_fields" is the ten pretrained fields. "errors_excluded" answers the
# question about the Errors? column. "behavioural_only" answers a second question the
# diagnostics raised on the first run: the vocabulary was fitted on pre-cut rows, so a
# post-cut calendar year is an unknown token on most scored rows, and a reviewer is right
# to ask how much of the surprise is that rather than behaviour. This set keeps the six
# fields that describe what the transaction DID and drops the calendar and the error flag.
FIELD_SETS = {
    "all_fields": None,   # every field
    "errors_excluded": [f for f in FIELDS if f != ERRORS_FIELD],
    "behavioural_only": ["hour", "amount_q", "use_chip", "mcc", "city", "state"],
}


# ------------------------------------------------------------------ loading ----

def load_inputs(prep_dir: str | Path) -> dict:
    """Load ONLY what the score needs. The fraud array is deliberately absent."""
    p = Path(prep_dir)
    meta = json.loads((p / "meta.json").read_text())
    assert list(meta["fields"]) == list(FIELDS), "field list drift between prep and code"
    d = {"meta": meta}
    for name in ("tokens", "user", "ts"):
        d[name] = np.load(p / f"{name}.npy")
    assert "fraud" not in d, "LABEL LEAK: fraud must not be in the scoring input dict"
    return d


# ------------------------------------------------------------------- scores ----

def pll_field_nll(model, cfg, tokens, rows, seg_start_of_row, device, batch_size, log_every=40):
    """Per-field masked-field NLL at the scored transaction.

    Returns nll [len(rows), F] float64. nll[i, f] = -log p(observed token of
    field f | the rest of that transaction and the account's window history),
    with field f of the scored position replaced by MASK.

    The window is left aligned exactly as embed.py builds them: positions
    0..L-1 hold the account's transactions ending AT the scored one, so the
    scored transaction sits at index L-1 and everything after is PAD.
    """
    import torch

    W = cfg["window"]
    F_ = tokens.shape[1]
    n = len(rows)
    nll = np.zeros((n, F_), dtype=np.float64)
    hist_len = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for c0 in range(0, n, batch_size):
        r = rows[c0:c0 + batch_size]
        hs = np.maximum(seg_start_of_row[r], r - W + 1)     # window start
        L = (r - hs + 1).astype(np.int64)                   # valid length, 1..W
        b = len(r)
        wins = np.full((b, W, F_), PAD_ID, dtype=np.int64)
        for j in range(b):
            wins[j, : L[j]] = tokens[hs[j]: r[j] + 1]
        pos = L - 1                                         # scored index in window
        hist_len[c0:c0 + b] = pos                           # transactions strictly before
        target = wins[np.arange(b), pos, :].copy()          # [b, F] observed tokens
        wt = torch.from_numpy(wins).to(device)
        pad = wt[:, :, 0] == PAD_ID
        ar = torch.arange(b, device=device)
        pt = torch.from_numpy(pos).to(device)
        for f in range(F_):
            inp = wt.clone()
            inp[ar, pt, f] = MASK_ID
            with torch.no_grad():
                h = model.encode(inp, pad)                  # [b, W, d]
                hs_sel = h[ar, pt]                          # [b, d]
                logits = model.field_logits(hs_sel, f).float()
                logp = torch.log_softmax(logits, dim=-1)
                tgt = torch.from_numpy(target[:, f]).to(device)
                nll[c0:c0 + b, f] = (-logp[ar, tgt]).double().cpu().numpy()
        if (c0 // batch_size) % log_every == 0:
            done = c0 + b
            rate = done / max(1e-9, time.time() - t0)
            print(f"[pll] {done}/{n} rows ({rate:.0f} rows/s)", flush=True)
    return nll, hist_len


def unigram_nll(tokens, rows, fit_mask, vocab_sizes):
    """NO MODEL. Per-field -log p(token) from marginal counts on the fit set.

    Add-one smoothing over the full per-field vocabulary, so an unseen token
    gets a finite, large surprise rather than an infinity.
    """
    F_ = tokens.shape[1]
    out = np.zeros((len(rows), F_), dtype=np.float64)
    obs = tokens[rows].astype(np.int64)
    for f in range(F_):
        V = int(vocab_sizes[f])
        col = tokens[:, f].astype(np.int64)
        cnt = np.bincount(col[fit_mask], minlength=V).astype(np.float64)
        p = (cnt + 1.0) / (cnt.sum() + V)
        out[:, f] = -np.log(p[np.clip(obs[:, f], 0, V - 1)])
    return out


def zsum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unweighted standardized sum. Uses no labels: the moments are over the
    scored rows themselves."""
    def z(x):
        s = x.std()
        return (x - x.mean()) / (s if s > 1e-12 else 1.0)
    return z(a) + z(b)


def build_scores(d, rows, ckpt_path, device_name, batch_size, fit_mask, threads):
    """LABEL-FREE BY CONSTRUCTION. No fraud array is reachable from here."""
    import torch
    from embed import load_model

    if threads:
        torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    device = torch.device(device_name)
    model, cfg = load_model(ckpt_path, d["meta"], device)

    seg_starts, seg_ends = user_segments(d["user"])
    seg_start_of_row = np.repeat(seg_starts, seg_ends - seg_starts)

    pll, hist_len = pll_field_nll(model, cfg, d["tokens"], rows, seg_start_of_row,
                                  device, batch_size)
    rar = unigram_nll(d["tokens"], rows, fit_mask, d["meta"]["vocab_sizes"])

    scores = {}
    for setname, fieldlist in FIELD_SETS.items():
        cols = (list(range(len(FIELDS))) if fieldlist is None
                else [FIELDS.index(f) for f in fieldlist])
        p = pll[:, cols].sum(axis=1)
        r = rar[:, cols].sum(axis=1)
        # the historical key for the all-fields combination has no suffix; keep it stable
        combo = "pll_plus_rarity" if setname == "all_fields" else f"pll_plus_rarity_{setname}"
        scores[f"pll_{setname}"] = p
        scores[f"rarity_{setname}"] = r
        scores[combo] = zsum(p, r)
    obs = d["tokens"][rows].astype(np.int64)
    diag = {
        "pll_field_mean_nll": {FIELDS[f]: float(pll[:, f].mean()) for f in range(len(FIELDS))},
        "rarity_field_mean_nll": {FIELDS[f]: float(rar[:, f].mean()) for f in range(len(FIELDS))},
        "mean_prior_transactions_in_window": float(hist_len.mean()),
        "share_of_scored_rows_with_no_prior_history": float((hist_len == 0).mean()),
        "share_of_scored_rows_with_an_out_of_vocabulary_value": float((obs == 1).any(axis=1).mean()),
        "out_of_vocabulary_share_by_field": {FIELDS[f]: float((obs[:, f] == 1).mean())
                                             for f in range(len(FIELDS))},
        "out_of_vocabulary_note": ("The vocabulary was built on pre-cut rows only, so a post-cut "
                                   "value never seen before maps to the unknown token. Both the "
                                   "model score and the rarity control assign such a value a "
                                   "high surprise, which is one of the things the control is "
                                   "there to separate."),
        "window": int(cfg["window"]),
        "mask_prob_pretrain": float(cfg["mask_prob"]),
        "d_model": int(cfg["d_model"]),
        "layers": int(cfg["layers"]),
    }
    return scores, diag


def score_sha256(scores: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(scores):
        h.update(k.encode())
        h.update(np.ascontiguousarray(scores[k], dtype=np.float64).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ metrics ----

def metric_vector(y, scores, names):
    """All reported metrics for one row selection, in a fixed order.

    ROC-AUC and PR-AUC come from scikit-learn so ties are handled the same way
    every other number on this page handles them. Recall at the top of the
    ranking reuses one descending argsort; ties there break in stable order.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        raise ValueError("degenerate")
    n = len(y)
    out = []
    for nm in names:
        s = scores[nm]
        out.append(roc_auc_score(y, s))
        out.append(average_precision_score(y, s))
        ys = y[np.argsort(-s, kind="stable")]
        cum = np.cumsum(ys)
        for fr in TOPK_FRACS:
            k = max(1, int(round(n * fr)))
            out.append(float(cum[k - 1]) / npos)
    return np.asarray(out, dtype=np.float64)


def metric_labels(names):
    lab = []
    for nm in names:
        lab.append((nm, "roc_auc"))
        lab.append((nm, "pr_auc"))
        for fr in TOPK_FRACS:
            lab.append((nm, f"recall_at_top_{fr:g}"))
    return lab


def cluster_bootstrap(user_of_row, y, scores, names, B, seed):
    """Entity-clustered bootstrap over test accounts, percentile 95%.

    One resample produces the WHOLE metric vector, so any difference between two
    scores is paired on identical rows by construction.
    """
    uniq, inv = np.unique(user_of_row, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    bnd = np.searchsorted(sorted_inv, np.arange(len(uniq)))
    bnd = np.append(bnd, len(sorted_inv))
    rows_by_user = [order[bnd[i]:bnd[i + 1]] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    m = len(metric_labels(names))
    out = np.full((B, m), np.nan)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_user[p] for p in pick])
        sub = {nm: scores[nm][idx] for nm in names}
        try:
            out[b] = metric_vector(y[idx], sub, names)
        except ValueError:
            pass
        if (b + 1) % 100 == 0:
            print(f"[boot] {b + 1}/{B}", flush=True)
    return out


def pct_ci(col):
    """Percentile 95% interval, or None when too many replicates were degenerate.

    A None here is a visible failure, never a silently dropped number."""
    ok = np.isfinite(col)
    if ok.sum() < 2:
        return None
    lo, hi = np.percentile(col[ok], 2.5), np.percentile(col[ok], 97.5)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return [float(lo), float(hi)]


# ---------------------------------------------------------------------- run ----

SCORE_NAMES = [
    "pll_all_fields",
    "rarity_all_fields",
    "pll_plus_rarity",
    "pll_errors_excluded",
    "rarity_errors_excluded",
    "pll_plus_rarity_errors_excluded",
    "pll_behavioural_only",
    "rarity_behavioural_only",
    "pll_plus_rarity_behavioural_only",
]

COMPARISONS = [
    ("pll_all_fields", "rarity_all_fields", "contextual surprise minus global rarity"),
    ("pll_plus_rarity", "rarity_all_fields", "the combined rule minus global rarity"),
    ("pll_plus_rarity", "pll_all_fields", "the combined rule minus contextual surprise"),
    ("pll_errors_excluded", "rarity_errors_excluded",
     "contextual surprise minus global rarity, errors field dropped from both"),
    ("pll_behavioural_only", "rarity_behavioural_only",
     "contextual surprise minus global rarity, behavioural fields only"),
    ("pll_plus_rarity_behavioural_only", "rarity_behavioural_only",
     "the combined rule minus global rarity, behavioural fields only"),
]


def run(args) -> dict:
    seed_everything(args.seed)
    d = load_inputs(args.prep)
    meta = d["meta"]
    ev = Path(args.eval)
    sel = np.load(ev / "select.npz")
    test_rows = np.sort(sel["test_rows"])
    test_users = set(sel["test_users"].tolist())

    # rarity fit set: PRE-cut rows of NON-test accounts, the same fit set the
    # shipped transfer evaluation uses for its frequency encodings.
    pre = d["ts"] < meta["cut_ts"]
    fit_mask = pre & ~np.isin(d["user"], list(test_users))

    if args.smoke:
        rng = np.random.default_rng(args.seed)
        test_rows = np.sort(rng.choice(test_rows, size=min(args.smoke, len(test_rows)),
                                       replace=False))

    print(f"[run] scoring {len(test_rows)} rows | rarity fit set {int(fit_mask.sum())} pre-cut rows",
          flush=True)

    # ---- SCORES FIRST. No label has been opened at this point. ----
    scores, diag = build_scores(d, test_rows, args.ckpt, args.device, args.batch_size,
                                fit_mask, args.threads)
    sha_before = score_sha256(scores)
    print(f"[run] scores built, sha256 {sha_before}", flush=True)

    # ---- LABELS ENTER HERE, AND ONLY HERE. ----
    y = np.load(Path(args.prep) / "fraud.npy")[test_rows].astype(np.int64)
    u = d["user"][test_rows]
    n_pos = int(y.sum())
    print(f"[run] labels loaded: {n_pos} positives in {len(y)} rows", flush=True)

    point = metric_vector(y, scores, SCORE_NAMES)
    labels = metric_labels(SCORE_NAMES)
    boot = cluster_bootstrap(u, y, scores, SCORE_NAMES, args.bootstrap, args.seed)

    by_score: dict[str, dict] = {nm: {} for nm in SCORE_NAMES}
    for i, (nm, met) in enumerate(labels):
        by_score[nm][met] = float(point[i])
        by_score[nm][met + "_ci"] = pct_ci(boot[:, i])

    idx_of = {(nm, met): i for i, (nm, met) in enumerate(labels)}
    comparisons, comparisons_by_key = [], {}
    for a, b, note in COMPARISONS:
        entry = {"a": a, "b": b, "note": note}
        for met in ("roc_auc", "pr_auc"):
            ia, ib = idx_of[(a, met)], idx_of[(b, met)]
            diff = float(point[ia] - point[ib])
            ci = pct_ci(boot[:, ia] - boot[:, ib])
            if ci is None:
                direction, excludes = "no_interval", False
            elif ci[0] > 0:
                direction, excludes = "a_wins", True
            elif ci[1] < 0:
                direction, excludes = "b_wins", True
            else:
                direction, excludes = "not_separated", False
            entry[met] = {"a_value": float(point[ia]), "b_value": float(point[ib]),
                          "difference": diff, "ci": ci,
                          "excludes_zero": excludes, "direction": direction}
        comparisons.append(entry)
        comparisons_by_key[f"{a}_vs_{b}"] = entry

    # errors-field diagnostic. Computed with labels and labelled as a diagnostic.
    from sklearn.metrics import roc_auc_score

    ei = FIELDS.index(ERRORS_FIELD)
    err_col = d["tokens"][:, ei].astype(np.int64)
    # the modal pre-cut value of the errors field is the "no error recorded" value.
    # derived from counts rather than assumed, since prep does not persist vocabs.
    modal_err_id = int(np.bincount(err_col[fit_mask],
                                   minlength=meta["vocab_sizes"][ei]).argmax())
    err_tok = err_col[test_rows]
    has_err = (err_tok != modal_err_id).astype(np.int64)
    err_diag = {
        "errors_field_note": ("TabFormer carries an 'Errors?' column. It is one of the ten "
                              "pretrained fields, so it enters the surprise score. Every score "
                              "is therefore reported a second time with that field dropped."),
        "modal_errors_token_id": modal_err_id,
        "share_of_scored_rows_with_an_error_flag": float(has_err.mean()),
        "fraud_rate_when_error_flag_set": (float(y[has_err == 1].mean())
                                           if has_err.sum() else None),
        "fraud_rate_when_no_error_flag": (float(y[has_err == 0].mean())
                                          if (1 - has_err).sum() else None),
        "roc_auc_of_the_error_flag_alone": float(roc_auc_score(y, has_err)),
    }

    # The other artifact a reviewer should be able to price: the vocabulary was fitted on
    # pre-cut rows, so a post-cut calendar year is an unknown token on most scored rows.
    # If that flag ranked fraud on its own, both scores would be reading the clock.
    yi = FIELDS.index("year")
    year_oov = (d["tokens"][test_rows, yi].astype(np.int64) == 1).astype(np.int64)
    if 0 < year_oov.sum() < len(year_oov):
        err_diag["roc_auc_of_the_unseen_year_flag_alone"] = float(roc_auc_score(y, year_oov))
        err_diag["fraud_rate_when_year_unseen"] = float(y[year_oov == 1].mean())
        err_diag["fraud_rate_when_year_seen"] = float(y[year_oov == 0].mean())
    err_diag["behavioural_only_note"] = (
        "The behavioural-only rows of the table drop the calendar fields and the error flag "
        "and keep the six fields that describe what the transaction did. That field set was "
        "chosen from the out-of-vocabulary diagnostic above, which carries no labels, and it "
        "ships in the same table as everything else whichever way it lands.")

    prevalence = n_pos / len(y)
    out = {
        "seed": args.seed,
        "versions": versions_dict(),
        "generated_by": "scripts/fm/protection_pll.py --check-able",
        "data_sources": meta["data_sources"],
        "labels": ["synthetic"],
        "what_this_is": (
            "Masked-field pseudo-log-likelihood read straight off the frozen backbone as a "
            "label-free behavioural surprise score, then measured against the real fraud "
            "labels on the same frozen evaluation split the shipped transfer table uses. A "
            "unigram rarity control built from marginal token counts with no model at all is "
            "scored the same way, because the honest question about any surprise score is how "
            "much of it is just globally rare values."),
        "method": {
            "objective": "masked-field, mask probability 0.15 at pretraining",
            "readout": ("one field of the scored transaction masked at a time, the negative "
                        "log-likelihood of the observed token summed over fields"),
            "fields": list(FIELDS),
            "window": diag["window"],
            "citation": "Salazar et al., ACL 2020, Masked Language Model Scoring",
            "full_mask_variant": ("NOT SHIPPED. Masking every field of a row at once is out of "
                                  "distribution for a 15% mask-probability pretrain. Any such "
                                  "variant would have to be labelled separately."),
            "retraining": "none. The checkpoint is frozen and used for inference only.",
            "checkpoint": "the full-corpus cardholder backbone (frozen, evaluation only)",
        },
        "zero_labels": {
            "fraud_labels_in_score_construction": False,
            "labels_loaded_after_scores": True,
            "score_sha256_before_labels": sha_before,
            "score_sha256_at_report": None,          # filled below
            "note": ("Enforced structurally. The scoring input dict never holds the fraud "
                     "array, build_scores() cannot reach it, and prep/fraud.npy is opened only "
                     "after every score exists and its sha256 is recorded. The hash is checked "
                     "again at reporting time, so the score bytes provably predate any label in "
                     "memory. Labels are used afterwards and only to measure how well the "
                     "ranking separates."),
        },
        "guards_constant": {
            "label_excluded_from_pretraining": bool(meta["leakage"]["label_excluded"]),
            "ids_excluded_from_vocab": bool(meta["leakage"]["ids_excluded"]),
            "corpus_time_truncated": True,
            "scored_rows_are_post_cut_and_never_pretrained_on": True,
            "note": ("The pretraining corpus is hard truncated before the cut date and the "
                     "scored rows are all post-cut, so no scored transaction was in the "
                     "pretraining data. The corpus does include pre-cut rows of the test "
                     "accounts, which is the same condition the shipped transfer table runs "
                     "under, and we state it rather than implying otherwise."),
        },
        "split": {
            "source": "the same frozen evaluation split as the shipped transfer table",
            "n_scored_rows": int(len(y)),
            "n_positives": n_pos,
            "positive_rate": prevalence,
            "n_test_accounts": int(len(np.unique(u))),
            "entity_disjoint": True,
            "note": json.loads((ev / "select_meta.json").read_text()),
        },
        "reference": {
            "random_ranking_pr_auc": prevalence,
            "random_ranking_roc_auc": 0.5,
            "note": ("A ranking with no information scores PR-AUC equal to the positive rate "
                     "and ROC-AUC 0.5. Read every number below against those two."),
        },
        "bootstrap": {"method": "entity-clustered bootstrap over test accounts",
                      "B": args.bootstrap, "ci": "percentile 95%",
                      "n_usable_replicates": int(np.isfinite(boot[:, 0]).sum()),
                      "paired": ("every score is measured on the same resampled rows inside one "
                                 "replicate, so the differences below are paired")},
        "scores": by_score,
        "comparisons": comparisons,
        "comparisons_by_key": comparisons_by_key,
        "diagnostics": {**diag, **err_diag},
        "check": {"command": "python scripts/fm/protection_pll.py --stage run ... --check",
                  "tolerance": 1e-6},
    }
    out["zero_labels"]["score_sha256_at_report"] = score_sha256(scores)
    assert out["zero_labels"]["score_sha256_at_report"] == sha_before, \
        "score bytes changed after labels were loaded"
    return out


# -------------------------------------------------------------------- check ----

def numeric_leaves(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(numeric_leaves(v, f"{prefix}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(numeric_leaves(v, f"{prefix}/{i}"))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if isinstance(obj, float) and not math.isfinite(obj):
            return out
        out[prefix] = float(obj)
    return out


def compare(fresh: dict, stored: dict, tol: float) -> int:
    a, b = numeric_leaves(fresh), numeric_leaves(stored)
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    bad = []
    for k in sorted(set(a) & set(b)):
        if abs(a[k] - b[k]) > tol:
            bad.append((k, b[k], a[k], abs(a[k] - b[k])))
    if only_a or only_b or bad:
        for k in only_a[:10]:
            print(f"CHECK: key only in recompute: {k}")
        for k in only_b[:10]:
            print(f"CHECK: key only in stored:    {k}")
        for k, s, f, dd in bad[:20]:
            print(f"CHECK FAILED {k}: stored {s!r} vs recomputed {f!r} (|diff| {dd:.3e})")
        print(f"CHECK FAILED: {len(bad)} mismatched, {len(only_a)} extra, {len(only_b)} missing")
        return 5
    print(f"CHECK OK: {len(a)} numeric leaves reproduce within {tol:g}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run"])
    ap.add_argument("--prep", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--smoke", type=int, default=0, help="score only N test rows (code path)")
    ap.add_argument("--check", action="store_true", help="recompute and compare at 1e-6; exit 0/5")
    ap.add_argument("--check-tol", type=float, default=1e-6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    fresh = run(args)
    if args.check:
        stored = json.loads(Path(args.out).read_text())
        return compare(fresh, stored, args.check_tol)
    atomic_write_json(args.out, fresh)
    print(f"[run] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
