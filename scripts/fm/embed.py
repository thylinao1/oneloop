"""embed.py: backbone feature extraction.

1) --merchants OUT.parquet : per-merchant embeddings = mean-pooled transaction
   encodings grouped by Merchant Name+City, PRE-cut transactions only
   (feeds WS-B3 whitespace).
2) --asof ROWS.npy --asof-out OUT.npy : as-of user embeddings for scored rows.
   LEAKAGE (c): each embedding encodes ONLY that user's transactions strictly
   before the scored transaction (a window ending right before it); the model
   never sees the scored transaction or anything after it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch

from common import PAD_ID, load_prep, seed_everything, user_segments
from model import build_model


@torch.no_grad()
def encode_batches(model, windows: np.ndarray, lengths: np.ndarray, device, batch_size: int):
    """windows [N,W,F] int64 (PAD-filled), lengths [N]. Yields (slice, h [b,W,d])."""
    for i in range(0, len(windows), batch_size):
        wb = torch.from_numpy(windows[i:i + batch_size]).to(device)
        pad = wb[:, :, 0] == PAD_ID
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h = model.encode(wb, pad)
        yield slice(i, i + len(wb)), h.float().cpu()


def load_model(ckpt_path: str, meta: dict, device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    model = build_model(meta, cfg["d_model"], cfg["layers"], cfg["heads"], cfg["ff"], cfg["window"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, cfg


def merchant_embeddings(model, cfg, d, device, out_path: str, batch_size: int):
    meta = d["meta"]
    W = cfg["window"]
    pre = d["ts"] < meta["cut_ts"]
    pre_idx = np.flatnonzero(pre)
    tokens, user, merch = d["tokens"][pre], d["user"][pre], d["merchant"][pre]
    seg_starts, seg_ends = user_segments(user)
    n_merch = int(d["merchant"].max()) + 1
    dsum = np.zeros((n_merch, model.d_model), dtype=np.float64)
    dcnt = np.zeros(n_merch, dtype=np.int64)

    # non-overlapping windows so each pre-cut txn is encoded exactly once
    starts, ends = [], []
    for s, e in zip(seg_starts, seg_ends):
        st = np.arange(s, e, W, dtype=np.int64)
        starts.append(st)
        ends.append(np.minimum(st + W, e))
    starts = np.concatenate(starts)
    ends = np.concatenate(ends)

    F_ = tokens.shape[1]
    chunk = 4096
    for c0 in range(0, len(starts), chunk):
        cs, ce = starts[c0:c0 + chunk], ends[c0:c0 + chunk]
        lens = ce - cs
        wins = np.full((len(cs), W, F_), PAD_ID, dtype=np.int64)
        rows = np.full((len(cs), W), -1, dtype=np.int64)
        for j, (s, e) in enumerate(zip(cs, ce)):
            wins[j, : e - s] = tokens[s:e]
            rows[j, : e - s] = np.arange(s, e)
        for sl, h in encode_batches(model, wins, lens, device, batch_size):
            r = rows[sl]
            valid = r >= 0
            enc = h.numpy()[valid]          # [n_txn, d]
            mids = merch[r[valid]]
            np.add.at(dsum, mids, enc)
            np.add.at(dcnt, mids, 1)
        if c0 % (chunk * 20) == 0:
            print(f"[embed] merchant pooling {c0}/{len(starts)} window-chunks", flush=True)

    keep = dcnt > 0
    emb = (dsum[keep] / dcnt[keep, None]).astype(np.float32)
    keys = pl.read_parquet(Path(d["prep_dir"]) / "merchant_keys.parquet")
    out = keys.filter(pl.Series(keep[keys["merchant_id"].to_numpy()])).with_columns(
        pl.Series("n_txns_pre_cut", dcnt[keep])
    )
    emb_df = pl.DataFrame({f"emb_{i}": emb[:, i] for i in range(emb.shape[1])})
    pl.concat([out, emb_df], how="horizontal").write_parquet(out_path)
    print(f"[embed] merchant embeddings: {int(keep.sum())} merchants (pre-cut pooled) -> {out_path}", flush=True)


def asof_embeddings(model, cfg, d, device, rows_path: str, out_path: str, batch_size: int):
    W = cfg["window"]
    rows = np.load(rows_path)
    tokens, user = d["tokens"], d["user"]
    seg_starts, seg_ends = user_segments(user)
    # per-row segment start
    seg_start_of_row = np.repeat(seg_starts, seg_ends - seg_starts)
    F_ = tokens.shape[1]
    out = np.zeros((len(rows), model.d_model + 1), dtype=np.float32)  # last col = has_history
    chunk = 4096
    for c0 in range(0, len(rows), chunk):
        r = rows[c0:c0 + chunk]
        hs = np.maximum(seg_start_of_row[r], r - W)  # history start
        lens = r - hs                                # strictly-before length (may be 0)
        wins = np.full((len(r), W, F_), PAD_ID, dtype=np.int64)
        for j, (s, e) in enumerate(zip(hs, r)):
            if e > s:
                wins[j, : e - s] = tokens[s:e]
        for sl, h in encode_batches(model, wins, lens, device, batch_size):
            hh = h.numpy()
            l = lens[sl]
            for j in range(len(hh)):
                lj = int(l[j])
                if lj > 0:
                    out[c0 + sl.start + j, : model.d_model] = hh[j, :lj].mean(axis=0)
                    out[c0 + sl.start + j, -1] = 1.0
        if c0 % (chunk * 20) == 0:
            print(f"[embed] as-of {c0}/{len(rows)} rows", flush=True)
    np.save(out_path, out)
    print(f"[embed] as-of embeddings for {len(rows)} scored rows -> {out_path} "
          f"({int(out[:, -1].sum())} with history)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--merchants", default="", help="output parquet for merchant embeddings")
    ap.add_argument("--asof", default="", help="npy of global row indices to embed as-of")
    ap.add_argument("--asof-out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    seed_everything(args.seed)
    d = load_prep(args.prep)
    d["prep_dir"] = args.prep
    device = torch.device(args.device)
    model, cfg = load_model(args.ckpt, d["meta"], device)
    if args.merchants:
        merchant_embeddings(model, cfg, d, device, args.merchants, args.batch_size)
    if args.asof:
        assert args.asof_out, "--asof requires --asof-out"
        asof_embeddings(model, cfg, d, device, args.asof, args.asof_out, args.batch_size)


if __name__ == "__main__":
    main()
