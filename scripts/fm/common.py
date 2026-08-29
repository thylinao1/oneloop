"""Shared constants + helpers for the One Loop backbone pipeline (WS-A).

LEAKAGE POLICY (hard requirements, enforced structurally here):
  (a) 'Is Fraud?' and label-derived fields NEVER enter model inputs/vocab.
  (b) pretraining corpus hard-truncated at cut_ts (recorded in every output).
  (c) as-of embeddings use ONLY transactions strictly before the scored one.
  (d) User/Card NEVER in the vocab; they index sequences only.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

# Model-input fields (tokenized). NOTE: no User, no Card, no 'Is Fraud?',
# no Merchant Name (merchants are represented by pooled encodings, not vocab ids).
FIELDS = [
    "year", "month", "day", "hour", "amount_q",
    "use_chip", "mcc", "city", "state", "errors",
]
FORBIDDEN_COLUMNS = {"User", "Card", "Is Fraud?"}

# Per-field special token ids (each field vocab reserves these).
PAD_ID = 0
UNK_ID = 1
MASK_ID = 2
N_SPECIAL = 3

TABFORMER_SHA256 = "e9f589a0958f40d60f81b1a2e8428db86e00c05755caf44fb055827976c0efa2"
TABFORMER_URL = "https://github.com/IBM/TabFormer (data/credit_card/transactions.tgz)"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def versions_dict() -> dict:
    import platform

    out = {"python": platform.python_version()}
    for lib in ("numpy", "polars", "pyarrow", "lightgbm", "sklearn", "torch"):
        try:
            mod = __import__(lib)
            out[lib] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    return out


def load_prep(prep_dir: str | Path) -> dict:
    """Load prep outputs into a dict of arrays + meta."""
    p = Path(prep_dir)
    meta = json.loads((p / "meta.json").read_text())
    assert set(meta["fields"]) == set(FIELDS), "field list drift between prep and code"
    for col in FORBIDDEN_COLUMNS:
        assert col not in meta["fields"], f"LEAKAGE: {col} in model fields"
    d = {"meta": meta}
    for name in ("tokens", "user", "ts", "fraud", "mcc_class", "amount", "merchant"):
        d[name] = np.load(p / f"{name}.npy")
    return d


def user_segments(user: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows are sorted by (user, ts). Return (seg_starts, seg_ends) per user run."""
    bounds = np.flatnonzero(np.diff(user)) + 1
    seg_starts = np.concatenate([[0], bounds])
    seg_ends = np.concatenate([bounds, [len(user)]])
    return seg_starts, seg_ends


def atomic_write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)
