"""pretrain.py: masked-field pretraining of the TabBERT backbone.

LEAKAGE (b): the corpus is HARD-TRUNCATED to rows with ts < cut_ts (from prep
meta). cut date recorded in every checkpoint + the summary JSON.

Checkpoints: latest.pt every --ckpt-minutes AND every epoch (epoch_N.pt, last 2
kept) to --ckpt dir. --resume auto restarts from latest.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import MASK_ID, PAD_ID, load_prep, seed_everything, user_segments, versions_dict
from model import build_model

LOG_EVERY = 50
CURVE_EVERY = 10


class WindowDataset(Dataset):
    """Sliding windows over per-user, time-sorted token rows (pre-cut only)."""

    def __init__(self, tokens: np.ndarray, user: np.ndarray, window: int, stride: int):
        self.tokens = tokens
        self.window = window
        seg_starts, seg_ends = user_segments(user)
        starts, ends = [], []
        for s, e in zip(seg_starts, seg_ends):
            if e - s < 2:
                continue
            st = np.arange(s, max(s + 1, e - window + 1), stride, dtype=np.int64)
            starts.append(st)
            ends.append(np.full(len(st), e, dtype=np.int64))
        self.starts = np.concatenate(starts) if starts else np.zeros(0, np.int64)
        self.ends = np.concatenate(ends) if ends else np.zeros(0, np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, k):
        s, e = self.starts[k], min(self.starts[k] + self.window, self.ends[k])
        w = np.full((self.window, self.tokens.shape[1]), PAD_ID, dtype=np.int64)
        w[: e - s] = self.tokens[s:e]
        return torch.from_numpy(w)


def save_ckpt(path: Path, model, opt, sched, epoch, step, loss_curve, config):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
        "epoch": epoch, "step": step, "loss_curve": loss_curve, "config": config,
    }, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=2048)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (0=off; smoke)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ckpt-minutes", type=float, default=30.0)
    ap.add_argument("--resume", default="auto", choices=["auto", "never"])
    ap.add_argument("--assert-improve", action="store_true",
                    help="smoke: hard-fail unless loss decreased >=10%% and is finite")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)  # NUS-COMPUTE.md Hopper cuDNN guard
    seed_everything(args.seed)

    d = load_prep(args.prep)
    meta = d["meta"]
    pre = d["ts"] < meta["cut_ts"]  # LEAKAGE (b): hard truncation
    tokens, user = d["tokens"][pre], d["user"][pre]
    print(f"[pretrain] corpus: {len(user)} pre-cut rows (cut={meta['corpus_cut_date']})", flush=True)

    ds = WindowDataset(tokens, user, args.window, args.stride)
    print(f"[pretrain] {len(ds)} windows (W={args.window}, stride={args.stride})", flush=True)
    device = torch.device(args.device)
    model = build_model(meta, args.d_model, args.layers, args.heads, args.ff, args.window).to(device)
    params_m = model.n_params() / 1e6
    print(f"[pretrain] model params: {params_m:.2f}M", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=gen,
                    num_workers=args.workers, drop_last=True, pin_memory=(device.type == "cuda"))
    steps_per_epoch = len(dl)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, t)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ckpt_dir = Path(args.ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    start_epoch, gstep, loss_curve = 0, 0, []
    config = {k: v for k, v in vars(args).items()}
    if args.resume == "auto" and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch, gstep, loss_curve = state["epoch"], state["step"], state["loss_curve"]
        print(f"[pretrain] RESUMED from epoch {start_epoch}, step {gstep}", flush=True)

    n_fields = tokens.shape[1]
    use_amp = device.type == "cuda"
    ema, first_ema = None, None
    last_ckpt_t = time.time()
    t0 = time.time()
    done = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch in dl:
            batch = batch.to(device, non_blocking=True)  # [B,W,F]
            pad = batch[:, :, 0] == PAD_ID  # [B,W]
            mask = (torch.rand(batch.shape, device=device) < args.mask_prob) & (~pad).unsqueeze(-1)
            if not mask.any():
                mask[0, 0, 0] = True
            inp = torch.where(mask, torch.full_like(batch, MASK_ID), batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                h = model.encode(inp, pad)
                loss_sum, n_masked = 0.0, 0
                for f in range(n_fields):
                    mf = mask[:, :, f]
                    if not mf.any():
                        continue
                    logits = model.field_logits(h[mf], f)
                    loss_sum = loss_sum + F.cross_entropy(logits.float(), batch[:, :, f][mf], reduction="sum")
                    n_masked += int(mf.sum())
                loss = loss_sum / max(1, n_masked)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            gstep += 1
            lval = float(loss.detach())
            assert math.isfinite(lval), f"NaN/Inf loss at step {gstep}"
            ema = lval if ema is None else 0.98 * ema + 0.02 * lval
            if first_ema is None and gstep >= 20:
                first_ema = ema
            if gstep % CURVE_EVERY == 0:
                loss_curve.append([gstep, round(ema, 5)])
            if gstep % LOG_EVERY == 0:
                sps = gstep / max(1e-9, time.time() - t0)
                print(f"[pretrain] epoch {epoch} step {gstep}/{total_steps} loss {lval:.4f} "
                      f"ema {ema:.4f} lr {sched.get_last_lr()[0]:.2e} {sps:.1f} steps/s", flush=True)
            if (time.time() - last_ckpt_t) / 60 >= args.ckpt_minutes:
                save_ckpt(latest, model, opt, sched, epoch, gstep, loss_curve, config)
                last_ckpt_t = time.time()
                print(f"[pretrain] checkpointed (30-min timer) at step {gstep}", flush=True)
            if args.max_steps and gstep >= args.max_steps:
                done = True
                break
        save_ckpt(latest, model, opt, sched, epoch + 1, gstep, loss_curve, config)
        save_ckpt(ckpt_dir / f"epoch_{epoch}.pt", model, opt, sched, epoch + 1, gstep, loss_curve, config)
        for old in sorted(ckpt_dir.glob("epoch_*.pt"))[:-2]:
            old.unlink()
        print(f"[pretrain] epoch {epoch} done at step {gstep}, ema {ema:.4f}", flush=True)
        if done:
            break

    save_ckpt(ckpt_dir / "final.pt", model, opt, sched, args.epochs, gstep, loss_curve, config)
    summary = {
        "params_m": round(params_m, 2),
        "epochs": args.epochs if not args.max_steps else f"partial({gstep} steps)",
        "steps": gstep,
        "loss_curve": loss_curve,
        "corpus_cut_date": meta["corpus_cut_date"],
        "corpus_rows_pre_cut": int(len(user)),
        "window": args.window, "stride": args.stride,
        "d_model": args.d_model, "layers": args.layers,
        "seed": args.seed,
        "versions": versions_dict(),
        "time_truncated": True,
    }
    (ckpt_dir / "pretrain_summary.json").write_text(json.dumps(summary, indent=1))
    ema_s = f"{ema:.4f}" if ema is not None else "n/a (no new steps; resumed at completion)"
    print(f"[pretrain] DONE steps={gstep} final ema={ema_s} first ema={first_ema}", flush=True)
    if args.assert_improve:
        assert ema is not None and first_ema is not None, "too few steps for smoke assert"
        assert ema < 0.9 * first_ema, f"SMOKE FAIL: loss did not decrease (first {first_ema:.4f} -> {ema:.4f})"
        print("[pretrain] smoke assert PASSED: loss decreased and finite", flush=True)


if __name__ == "__main__":
    main()
