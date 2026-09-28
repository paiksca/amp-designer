"""A small property-conditioned peptide language model.

The model is a decoder-only transformer over the 20 amino acids. Each sequence is
prefixed with four control tokens describing length, net charge, predicted Gram-
negative potency, and predicted hemolytic safety. Conditioning on properties is
what lets sampling reproduce the reference AMP property distributions by
construction, not by rejection, which is one of the four Phase-1 criteria.

Predicted potency and safety come from this project's own MIC and hemolysis
ensembles, so the control signal is self-distilled from public assay data.

Attention keeps a key/value cache, because generating a few hundred thousand
sequences without one costs roughly the sequence length in wasted compute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AA = "ACDEFGHIKLMNPQRSTVWY"
PAD, BOS, EOS = 20, 21, 22
N_BASE = 23

LEN_EDGES = [12, 16, 20, 25, 32, 40]            # 7 bins
CHARGE_EDGES = [0.5, 2.5, 4.5, 6.5, 8.5, 11.5]  # 7 bins
POTENCY_BINS = 5
SAFETY_BINS = 4

LEN_OFFSET = N_BASE
CHARGE_OFFSET = LEN_OFFSET + len(LEN_EDGES) + 1
POTENCY_OFFSET = CHARGE_OFFSET + len(CHARGE_EDGES) + 1
SAFETY_OFFSET = POTENCY_OFFSET + POTENCY_BINS
VOCAB = SAFETY_OFFSET + SAFETY_BINS

N_CONTROL = 4
MAX_LEN = 50
MIN_LEN = 8
BLOCK = 1 + N_CONTROL + MAX_LEN + 1


def bin_of(value: float, edges: list[float]) -> int:
    i = 0
    for e in edges:
        if value > e:
            i += 1
    return i


def quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    ranks = values.argsort(kind="stable").argsort(kind="stable")
    return np.minimum((ranks * n_bins) // len(values), n_bins - 1)


@dataclass
class Config:
    n_layer: int = 4
    n_head: int = 6
    d_model: int = 192
    dropout: float = 0.1
    vocab: int = VOCAB
    block: int = BLOCK


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x, cache: list | None = None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.d_head).transpose(1, 2)

        if cache is not None:
            if cache:
                k = torch.cat([cache[0], k], dim=2)
                v = torch.cat([cache[1], v], dim=2)
            cache[:] = [k, v]
            # One new query attends to the whole cached prefix, so no mask is needed.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


class PeptideLM(nn.Module):
    def __init__(self, cfg: Config | None = None):
        super().__init__()
        cfg = cfg or Config()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.block, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, caches=None, offset: int = 0):
        b, t = idx.shape
        pos = torch.arange(offset, offset + t, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if caches is None else caches[i])
        return self.head(self.ln_f(x))


def encode_batch(sequences: list[str], controls: np.ndarray) -> torch.Tensor:
    n = len(sequences)
    out = np.full((n, BLOCK), PAD, dtype=np.int64)
    out[:, 0] = BOS
    out[:, 1] = LEN_OFFSET + controls[:, 0]
    out[:, 2] = CHARGE_OFFSET + controls[:, 1]
    out[:, 3] = POTENCY_OFFSET + controls[:, 2]
    out[:, 4] = SAFETY_OFFSET + controls[:, 3]
    for i, s in enumerate(sequences):
        for j, c in enumerate(s):
            out[i, 5 + j] = AA.index(c)
        out[i, 5 + len(s)] = EOS
    return torch.from_numpy(out)


def control_tokens(controls: np.ndarray) -> torch.Tensor:
    n = controls.shape[0]
    idx = torch.empty((n, 1 + N_CONTROL), dtype=torch.long)
    idx[:, 0] = BOS
    idx[:, 1] = torch.from_numpy(LEN_OFFSET + controls[:, 0])
    idx[:, 2] = torch.from_numpy(CHARGE_OFFSET + controls[:, 1])
    idx[:, 3] = torch.from_numpy(POTENCY_OFFSET + controls[:, 2])
    idx[:, 4] = torch.from_numpy(SAFETY_OFFSET + controls[:, 3])
    return idx


@torch.no_grad()
def sample(
    model: PeptideLM,
    controls: np.ndarray,
    generator: torch.Generator,
    temperature: float = 1.0,
    top_p: float = 0.95,
    min_len: int = MIN_LEN,
    max_len: int = MAX_LEN,
) -> list[str]:
    """Nucleus sampling with a key/value cache, one sequence per control row."""
    model.eval()
    n = controls.shape[0]
    idx = control_tokens(controls)
    caches: list[list] = [[] for _ in range(model.cfg.n_layer)]

    logits = model(idx, caches=caches, offset=0)[:, -1, :].float()
    offset = idx.shape[1]

    # Rows that have emitted EOS are dropped from the batch, not carried to
    # the end on padding. Most peptides finish well before 50 residues, so keeping
    # them would roughly double the work.
    live = torch.arange(n)
    letters = np.full((n, max_len), -1, dtype=np.int8)
    lengths = np.zeros(n, dtype=np.int32)

    for step in range(max_len + 1):
        logits[:, PAD] = -float("inf")
        logits[:, BOS] = -float("inf")
        logits[:, N_BASE:] = -float("inf")
        if step < min_len:
            logits[:, EOS] = -float("inf")
        if step == max_len:
            logits[:, :20] = -float("inf")

        probs = F.softmax(logits / temperature, dim=-1)
        if top_p < 1.0:
            srt, order = torch.sort(probs, descending=True, dim=-1)
            cum = srt.cumsum(dim=-1)
            srt = srt.masked_fill(cum - srt > top_p, 0.0)
            srt = srt / srt.sum(dim=-1, keepdim=True)
            nxt = order.gather(-1, torch.multinomial(srt, 1, generator=generator))
        else:
            nxt = torch.multinomial(probs, 1, generator=generator)
        nxt = nxt.squeeze(1)

        emitted = nxt < 20
        if bool(emitted.any()):
            rows = live[emitted].numpy()
            letters[rows, lengths[rows]] = nxt[emitted].numpy().astype(np.int8)
            lengths[rows] += 1

        keep = torch.nonzero(emitted).squeeze(1)
        if keep.numel() == 0:
            break
        live = live[keep]
        nxt = nxt[keep]
        for cache in caches:
            cache[0] = cache[0][keep]
            cache[1] = cache[1][keep]

        logits = model(nxt[:, None], caches=caches, offset=offset)[:, -1, :].float()
        offset += 1

    out = []
    for i in range(n):
        out.append("".join(AA[c] for c in letters[i, : lengths[i]]))
    return out


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
