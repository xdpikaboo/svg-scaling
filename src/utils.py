"""Utilities shared across phases: seeding, parameter counting, LR schedule,
and an incrementally-flushed metrics logger that writes the JSON shape from
the project spec"""

from __future__ import annotations

import json
import math
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Not strict determinism: we accept some non-determinism
    from cuDNN/SDPA in exchange for throughput. ``torch.use_deterministic_algorithms``
    is intentionally not enabled.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module, exclude_embedding: bool = False) -> int:
    """Count trainable parameters in ``model``.

    When ``exclude_embedding`` is True, subtracts every parameter belonging to
    an ``nn.Embedding`` submodule. The LM head, when weight-tied to the token
    embedding, shares storage and is naturally counted once; when untied (µP's
    ``MuReadout``) it lives in a separate ``nn.Linear`` and remains counted.
    This matches the ``n_params_non_embedding`` field of the run JSON.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not exclude_embedding:
        return total
    embedding_numel = 0
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    embedding_numel += p.numel()
    return total - embedding_numel


def cosine_lr_with_warmup(
    step: int,
    max_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup to ``max_lr``, then cosine decay to ``max_lr * min_lr_ratio``.

    - ``step < warmup_steps``: linear ramp from 0 to ``max_lr``.
    - ``warmup_steps <= step <= max_steps``: cosine from ``max_lr`` to
      ``max_lr * min_lr_ratio``.
    - ``step > max_steps``: clamped at ``max_lr * min_lr_ratio``.
    """
    min_lr = max_lr * min_lr_ratio
    if step < warmup_steps:
        # warmup_steps assumed >= 1 by callers; guard anyway.
        if warmup_steps <= 0:
            return max_lr
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_steps = max(1, max_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


class MetricsLogger:
    """Incrementally writes the run JSON shape from the project spec

    Atomic flush via temp-file + ``os.replace`` after every update, so a crash
    leaves a valid (if partial) JSON file on disk. The single source of truth
    for ``analysis.ipynb``, no other format.

    Convention: ``tokens_seen`` passed to ``log_step`` is *cumulative* at the
    end of the step. ``tokens_per_second_avg`` at finalize is computed from
    that cumulative count divided by ``wall_seconds``.
    """

    def __init__(
        self,
        path: str,
        run_id: str,
        config_path: str,
        config_resolved: dict[str, Any],
        git_sha: str,
    ) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._t0 = time.monotonic()
        self._data: dict[str, Any] = {
            "run_id": run_id,
            "config_path": config_path,
            "config_resolved": config_resolved,
            "n_params": None,
            "n_params_non_embedding": None,
            "tokens_seen_per_step": [],
            "train_loss": [],
            "val_loss": [],
            "val_loss_final": None,
            "wall_seconds": None,
            "tokens_per_second_avg": None,
            "peak_gpu_mem_bytes": None,
            "git_sha": git_sha,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "status": "running",
        }
        self._flush()

    def set_param_counts(self, n_params: int, n_params_non_embedding: int) -> None:
        self._data["n_params"] = int(n_params)
        self._data["n_params_non_embedding"] = int(n_params_non_embedding)
        self._flush()

    def log_step(
        self,
        tokens_seen: int,
        train_loss: float,
        val_loss: float | None = None,
    ) -> None:
        self._data["tokens_seen_per_step"].append(int(tokens_seen))
        self._data["train_loss"].append(float(train_loss))
        self._data["val_loss"].append(None if val_loss is None else float(val_loss))
        self._flush()

    def set_status(self, status: str) -> None:
        # "ok" | "oom" | "diverged" | "crashed" (running while in-flight)
        self._data["status"] = status
        self._flush()

    def set_peak_gpu_mem(self, n_bytes: int) -> None:
        self._data["peak_gpu_mem_bytes"] = int(n_bytes)
        self._flush()

    def finalize(self, val_loss_final: float | None = None) -> None:
        self._data["wall_seconds"] = float(time.monotonic() - self._t0)
        self._data["finished_at"] = datetime.now(timezone.utc).isoformat()
        if val_loss_final is not None:
            self._data["val_loss_final"] = float(val_loss_final)
        elif self._data["val_loss"]:
            # Best-effort: use the last non-None val_loss seen.
            non_null = [v for v in self._data["val_loss"] if v is not None]
            if non_null:
                self._data["val_loss_final"] = non_null[-1]
        toks = self._data["tokens_seen_per_step"]
        if toks and self._data["wall_seconds"] and self._data["wall_seconds"] > 0:
            self._data["tokens_per_second_avg"] = toks[-1] / self._data["wall_seconds"]
        if self._data["status"] == "running":
            self._data["status"] = "ok"
        self._flush()

    def _flush(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)
