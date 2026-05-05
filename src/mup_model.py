"""muP variant of GPT.

The five differences from src/model.py are listed inline below. Each one is
a documented common bug in the muP literature.

Build the model via ``build_mup_gpt(cfg)``, NOT by instantiating MuGPT
directly. The factory builds the throwaway base/delta models that
``mup.set_base_shapes`` needs to register width-scaling info.
"""

from __future__ import annotations

import math
from dataclasses import replace

import mup
import torch
import torch.nn as nn
import torch.nn.functional as F
from mup import MuReadout, set_base_shapes

from src.config import ModelConfig


# ---------------------------------------------------------------------------
# Model components, same shape as src/model.py, two muP-specific differences:
#   (1) attention scale 1/d_head instead of 1/sqrt(d_head)
#   (2) head is a MuReadout, untied from the embedding
# ---------------------------------------------------------------------------

class MuCausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, (
            f"d_model ({cfg.d_model}) must be divisible by n_head ({cfg.n_head})"
        )
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout_p = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        # muP attention scale is 1/d_head, not 1/sqrt(d_head).
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=1.0 / self.d_head,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MuMLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class MuBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = MuCausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = MuMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MuGPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MuBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        # MuReadout is untied from tok_emb. Don't reassign weight!
        self.head = MuReadout(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        """Returns logits if ``targets is None``, else ``(logits, loss)``.

        The single-tensor return path is what mup's coord-check expects (its
        per-module forward hook iterates tuples and chokes on the ``None``
        loss element). Train.py always passes targets, so it gets the tuple.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        return logits, loss


# ---------------------------------------------------------------------------
# Factory: build target + base + delta and register shapes
# ---------------------------------------------------------------------------

# Base/delta are throwaway, only used for shape registration.
_BASE_D_MODEL = 64
_DELTA_D_MODEL = 128
# Both base and delta use n_head=4 so d_head ∈ {16, 32}, divisible.
_BASE_DELTA_N_HEAD = 4


def _make_base_delta(target_cfg: ModelConfig) -> tuple[MuGPT, MuGPT]:
    """Build base (d=64) + delta (d=128) models with the same n_layer as target."""
    base_cfg = replace(
        target_cfg,
        d_model=_BASE_D_MODEL,
        d_ff=_BASE_D_MODEL * 4,
        n_head=_BASE_DELTA_N_HEAD,
    )
    delta_cfg = replace(
        target_cfg,
        d_model=_DELTA_D_MODEL,
        d_ff=_DELTA_D_MODEL * 4,
        n_head=_BASE_DELTA_N_HEAD,
    )
    return MuGPT(base_cfg), MuGPT(delta_cfg)


def _mup_init(model: nn.Module, n_layer: int) -> None:
    """Apply µP-aware init. Per the mup transformer README + GPT-2 residual trick."""
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias"):
            nn.init.zeros_(p)
            continue
        if "ln" in name or "norm" in name:
            # LayerNorm gamma stays at default 1 (default init from nn.LayerNorm).
            continue
        if "tok_emb" in name or "pos_emb" in name:
            # Input embeddings: standard init, no width scaling needed.
            mup.init.normal_(p, mean=0.0, std=1.0)
        elif "head" in name:
            # MuReadout has its own init machinery, leave alone.
            continue
        elif name.endswith("attn.proj.weight") or name.endswith("mlp.proj.weight"):
            # GPT-2 residual scaling, mup-aware.
            mup.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        elif name.endswith(".weight"):
            mup.init.normal_(p, mean=0.0, std=0.02)


def build_mup_gpt(cfg: ModelConfig) -> MuGPT:
    """Build a muP-configured GPT.

    Steps in order (the order matters):
      1. Construct target.
      2. Construct base + delta (different widths, same n_layer).
      3. mup.set_base_shapes(target, base, delta=delta) to register
         ``infshape`` on every target param.
      4. Apply muP-aware init.

    The base/delta models are discarded after step 3; they're only used for
    shape registration.
    """
    target = MuGPT(cfg)
    base, delta = _make_base_delta(cfg)
    set_base_shapes(target, base, delta=delta)
    _mup_init(target, n_layer=cfg.n_layer)
    return target
