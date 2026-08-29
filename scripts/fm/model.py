"""model.py: TabBERT-style masked-field transformer over transaction windows."""
from __future__ import annotations

import torch
import torch.nn as nn


class TabBERT(nn.Module):
    def __init__(self, vocab_sizes: list[int], d_model: int = 512, n_layers: int = 6,
                 n_heads: int = 8, ff: int = 2048, window: int = 16, dropout: float = 0.1):
        super().__init__()
        self.vocab_sizes = list(vocab_sizes)
        self.window = window
        self.d_model = d_model
        self.field_emb = nn.ModuleList([nn.Embedding(v, d_model, padding_idx=0) for v in vocab_sizes])
        self.pos_emb = nn.Embedding(window, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList([nn.Linear(d_model, v) for v in vocab_sizes])

    def encode(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """tokens [B,W,F] long; pad_mask [B,W] True where PAD. Returns [B,W,d]."""
        x = self.field_emb[0](tokens[:, :, 0])
        for i in range(1, len(self.field_emb)):
            x = x + self.field_emb[i](tokens[:, :, i])
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = x + self.pos_emb(pos)[None, :, :]
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.norm(h)

    def field_logits(self, h: torch.Tensor, field: int) -> torch.Tensor:
        return self.heads[field](h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(meta: dict, d_model: int, n_layers: int, n_heads: int, ff: int,
                window: int, dropout: float = 0.1) -> TabBERT:
    return TabBERT(meta["vocab_sizes"], d_model, n_layers, n_heads, ff, window, dropout)
