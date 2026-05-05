"""Composable training-config dataclasses + a small YAML loader.

The five size-specific YAMLs (configs/sp_*.yaml, configs/mup_*.yaml) ``extends:``
configs/base.yaml and override only what differs. The loader resolves the
``extends:`` chain (with cycle detection) and deep-merges into a TrainConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 4096
    block_size: int = 1024
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.0
    bias: bool = False


@dataclass
class OptimConfig:
    name: str = "adamw"
    lr: float | None = None  # None until set by Phase 6 LR sweep
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    eps: float = 1.0e-8


@dataclass
class ScheduleConfig:
    warmup_steps: int = 10
    max_steps: int = 92
    min_lr_ratio: float = 0.1


@dataclass
class BatchConfig:
    tokens_per_step: int = 524288
    block_size: int = 1024
    micro_batch: int | None = None  # set per-size

    @property
    def grad_accum(self) -> int:
        if self.micro_batch is None:
            raise ValueError("BatchConfig.micro_batch is None, set it in the size config")
        n = self.tokens_per_step // (self.micro_batch * self.block_size)
        if n * self.micro_batch * self.block_size != self.tokens_per_step:
            raise ValueError(
                f"tokens_per_step ({self.tokens_per_step}) must equal "
                f"micro_batch ({self.micro_batch}) * block_size ({self.block_size}) * grad_accum"
            )
        return n


@dataclass
class EvalConfig:
    every_steps: int = 10
    val_tokens: int = 524288


@dataclass
class PathsConfig:
    train_bin: str = "data/tokens/train.bin"
    val_bin: str = "data/tokens/val.bin"
    tokenizer: str = "data/tokenizer/tokenizer.json"
    checkpoints_dir: str = "checkpoints"
    runs_dir: str = "runs"


@dataclass
class TrainConfig:
    seed: int = 0
    name: str = "unnamed"
    parameterization: str = "sp"  # "sp" or "mup"
    precision: str = "bfloat16"
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# ---------------------------------------------------------------------------
# YAML loader with extends + deep merge
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge: ``over`` wins on scalars, dicts merge in place."""
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_with_extends(path: Path, _seen: set[Path] | None = None) -> dict:
    if _seen is None:
        _seen = set()
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"circular extends involving {path}")
    _seen.add(path)
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    parent_rel = d.pop("extends", None)
    if parent_rel:
        parent = (path.parent / parent_rel).resolve()
        parent_d = _load_with_extends(parent, _seen)
        d = _deep_merge(parent_d, d)
    return d


def _coerce_field(target_type: Any, value: Any) -> Any:
    """Coerce YAML scalars into expected types where it matters (tuples)."""
    # tuple type hints from typing/PEP 604 are messy at runtime; just do best-effort.
    type_str = str(target_type)
    if "tuple" in type_str.lower() and isinstance(value, list):
        return tuple(value)
    return value


def _build_dataclass(cls: type, data: dict) -> Any:
    """Construct a dataclass from a dict, recursing into nested dataclass fields."""
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        if is_dataclass(f.type):
            # When forward refs are resolved, f.type is a class; otherwise a string.
            kwargs[f.name] = _build_dataclass(f.type, val) if isinstance(val, dict) else val
        else:
            # Resolve nested dataclasses via name lookup for string-typed fields.
            t = _resolve_nested_type(cls, f.name)
            if t is not None and isinstance(val, dict):
                kwargs[f.name] = _build_dataclass(t, val)
            else:
                kwargs[f.name] = _coerce_field(f.type, val)
    return cls(**kwargs)


_NESTED_DATACLASSES = {
    "model": ModelConfig,
    "optimizer": OptimConfig,
    "schedule": ScheduleConfig,
    "batch": BatchConfig,
    "eval": EvalConfig,
    "paths": PathsConfig,
}


def _resolve_nested_type(cls: type, field_name: str) -> type | None:
    if cls is TrainConfig and field_name in _NESTED_DATACLASSES:
        return _NESTED_DATACLASSES[field_name]
    return None


def load_config(path: str | Path) -> TrainConfig:
    """Load a YAML config (with optional ``extends:``) into a TrainConfig."""
    raw = _load_with_extends(Path(path))
    return _build_dataclass(TrainConfig, raw)


def apply_overrides(cfg: TrainConfig, overrides: list[str]) -> TrainConfig:
    """CLI override shape: ['optimizer.lr=3e-3', 'batch.micro_batch=16'].

    Applied in-place on a deep copy of ``cfg``. Type-coerces leaf values as
    int / float / bool / str using YAML scalar parsing for consistency.
    """
    import copy
    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override '{ov}' must be of form key=value")
        key, raw = ov.split("=", 1)
        # YAML scalar parse so 'true'/'16'/'[1,2]' work; then patch the YAML 1.1
        # quirk where '3e-3' (no decimal) is returned as a string.
        val = yaml.safe_load(raw)
        if isinstance(val, str):
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass  # genuine string
        obj: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)
    return cfg
