"""Training loop. Trains one config to completion, evaluating every 10 steps
and writing the metrics JSON incrementally.

CLI:
    python -m src.train --config configs/sp_tiny.yaml --override optimizer.lr=3e-3
    python -m src.train --config configs/sp_tiny.yaml --max-steps 20 --dry-run-profile

Failure modes:
    OOM         -> status="oom",       exit 1
    NaN/Inf     -> status="diverged",  exit 1
    KeyboardInt -> status="crashed",   exit 130

The sweep wrapper catches non-zero exits and continues to the next config.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import TrainConfig, apply_overrides, load_config
from src.model import GPT
from src.utils import MetricsLogger, cosine_lr_with_warmup, count_params, set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def make_run_id(cfg: TrainConfig) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.optimizer.lr is None:
        lr_str = "lrnone"
    else:
        # 3e-3 style; strip leading zero in exponent.
        s = f"{cfg.optimizer.lr:.0e}"
        s = s.replace("e-0", "e-").replace("e+0", "e+")
        lr_str = f"lr{s}"
    return f"{cfg.parameterization}_{cfg.name.split('_', 1)[-1]}_{lr_str}_{ts}"


def pick_device_and_dtype(cfg_precision: str) -> tuple[torch.device, torch.dtype, bool]:
    """Returns (device, autocast_dtype, use_grad_scaler)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # bf16 path on Ampere+; fp16 + GradScaler on Turing or older.
        if cfg_precision == "bfloat16" and torch.cuda.is_bf16_supported():
            return device, torch.bfloat16, False
        return device, torch.float16, True
    # CPU smoke: fp32, no autocast.
    return torch.device("cpu"), torch.float32, False


def load_mmap(path: Path) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(f"token bin not found: {path}")
    return np.memmap(path, mode="r", dtype=np.uint16)


def get_batch(
    data: np.memmap,
    micro_batch: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random offsets and return (x, y) of shape (micro_batch, block_size)."""
    high = len(data) - block_size - 1
    if high <= 0:
        raise ValueError(
            f"data of length {len(data)} too small for block_size {block_size}"
        )
    ix = torch.randint(0, high, (micro_batch,))
    # uint16 -> int64 because nn.Embedding needs int64.
    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64))
        for i in ix.tolist()
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
        for i in ix.tolist()
    ])
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def configure_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    device: torch.device,
    parameterization: str = "sp",
) -> torch.optim.Optimizer:
    """Two param groups: weights (decay) vs biases/layernorms (no decay).

    For ``parameterization == "mup"`` returns ``mup.MuAdamW`` so the µP
    per-parameter LR multipliers (set by ``mup.set_base_shapes``) are
    respected, using regular AdamW silently undoes µP (the project spec).
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    use_fused = device.type == "cuda"
    if parameterization == "mup":
        from mup import MuAdamW
        # MuAdamW wraps torch.optim.AdamW; accepts the same kwargs.
        try:
            return MuAdamW(groups, lr=lr, betas=betas, eps=eps, fused=use_fused)
        except TypeError:
            # Older mup versions don't pass through ``fused`` cleanly.
            return MuAdamW(groups, lr=lr, betas=betas, eps=eps)
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=use_fused)


@torch.no_grad()
def evaluate(
    model: GPT,
    val_data: np.memmap,
    micro_batch: int,
    block_size: int,
    eval_tokens: int,
    device: torch.device,
    autocast_ctx,
) -> float:
    """Mean cross-entropy loss over ~eval_tokens worth of random val batches."""
    model.eval()
    n_batches = max(1, eval_tokens // (micro_batch * block_size))
    losses: list[float] = []
    for _ in range(n_batches):
        x, y = get_batch(val_data, micro_batch, block_size, device)
        with autocast_ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Dry-run projection
# ---------------------------------------------------------------------------

# the project spec size table, used for cross-size FLOPs scaling in dry-run.
_SIZE_TABLE = {
    "sp_tiny":   {"n_params": 1.05e6,  "n_layer": 2,  "d_model": 128},
    "sp_small":  {"n_params": 2.75e6,  "n_layer": 4,  "d_model": 192},
    "sp_medium": {"n_params": 6.0e6,   "n_layer": 6,  "d_model": 256},
    "sp_large":  {"n_params": 16.1e6,  "n_layer": 8,  "d_model": 384},
    "sp_xl":     {"n_params": 34.1e6,  "n_layer": 10, "d_model": 512},
}


def project_wall_times(
    measured_size: str,
    measured_tokens_per_sec: float,
    full_max_steps: int,
    tokens_per_step: int,
) -> dict[str, float]:
    """Project full-run wall time for each size assuming throughput ~ 1/N.

    Rough approximation: training FLOPs is about 6 * N * tokens_seen, so
    tokens/sec scales as 1/N on a fixed device. Good to about 30% accuracy
    for cross-size projection.
    """
    if measured_size not in _SIZE_TABLE:
        return {}
    n_measured = _SIZE_TABLE[measured_size]["n_params"]
    out: dict[str, float] = {}
    total_tokens = full_max_steps * tokens_per_step
    for name, info in _SIZE_TABLE.items():
        scale = n_measured / info["n_params"]
        tps = measured_tokens_per_sec * scale
        out[name] = total_tokens / max(1.0, tps)  # seconds
    return out


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig, args: argparse.Namespace) -> int:
    # Validate config.
    if cfg.optimizer.lr is None:
        print("ERROR: optimizer.lr is None, pass --override optimizer.lr=X "
              "or fill in the YAML.", file=sys.stderr)
        return 2

    # Override max_steps if provided. Cosine schedule denominator follows.
    if args.max_steps is not None:
        cfg.schedule.max_steps = args.max_steps

    set_seed(cfg.seed)
    device, autocast_dtype, use_grad_scaler = pick_device_and_dtype(cfg.precision)
    print(f"device={device}  autocast_dtype={autocast_dtype}  grad_scaler={use_grad_scaler}")

    # Run id, paths, metrics file.
    is_dry_run = args.dry_run_profile
    run_id = make_run_id(cfg)
    if is_dry_run:
        run_id = f"dryrun_{run_id}"
    runs_dir = Path(cfg.paths.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = runs_dir / f"{run_id}.json"
    print(f"run_id={run_id}")
    print(f"metrics={metrics_path}")

    # Load token bins.
    train_data = load_mmap(Path(cfg.paths.train_bin))
    val_data = load_mmap(Path(cfg.paths.val_bin))
    print(f"train.bin: {len(train_data):,} tokens   val.bin: {len(val_data):,} tokens")

    # Build model, dispatch on parameterization (the project spec).
    if cfg.parameterization == "mup":
        from src.mup_model import build_mup_gpt
        model = build_mup_gpt(cfg.model).to(device)
    elif cfg.parameterization == "sp":
        model = GPT(cfg.model).to(device)
    else:
        raise ValueError(f"unknown parameterization: {cfg.parameterization!r}")
    n_params = count_params(model)
    n_params_non_emb = count_params(model, exclude_embedding=True)
    print(f"model={cfg.name}  n_params={n_params:,}  non_emb={n_params_non_emb:,}  "
          f"parameterization={cfg.parameterization}")

    # Optimizer + grad accum. configure_optimizer dispatches on
    # parameterization to use mup.MuAdamW when needed.
    grad_accum = cfg.batch.grad_accum  # validated on access
    optimizer = configure_optimizer(
        model, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas, eps=cfg.optimizer.eps, device=device,
        parameterization=cfg.parameterization,
    )
    scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None
    if device.type == "cuda" and autocast_dtype != torch.float32:
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
    else:
        # No-op context manager for CPU/fp32.
        from contextlib import nullcontext
        autocast_ctx = nullcontext()

    # Metrics file.
    metrics = MetricsLogger(
        path=str(metrics_path),
        run_id=run_id,
        config_path=str(args.config),
        config_resolved=asdict(cfg),
        git_sha=get_git_sha(),
    )
    metrics.set_param_counts(n_params, n_params_non_emb)
    if is_dry_run:
        metrics.set_status("dry_run")

    # Training loop.
    max_steps = cfg.schedule.max_steps
    warmup_steps = cfg.schedule.warmup_steps
    max_lr = cfg.optimizer.lr
    min_lr_ratio = cfg.schedule.min_lr_ratio
    micro_batch = cfg.batch.micro_batch
    block_size = cfg.batch.block_size
    eval_every = cfg.eval.every_steps
    eval_tokens = cfg.eval.val_tokens
    tokens_per_step = cfg.batch.tokens_per_step

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"\nstarting: max_steps={max_steps}  warmup={warmup_steps}  "
          f"micro_batch={micro_batch}  grad_accum={grad_accum}  "
          f"tokens/step={tokens_per_step:,}")

    t_start = time.monotonic()
    tokens_seen = 0
    last_val_loss: float | None = None
    last_step_time: float | None = None

    try:
        for step in range(max_steps):
            # LR for this step.
            lr = cosine_lr_with_warmup(step, max_steps, warmup_steps, max_lr, min_lr_ratio)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Grad accum.
            optimizer.zero_grad(set_to_none=True)
            train_loss_accum = 0.0
            t_step = time.monotonic()
            for _ in range(grad_accum):
                x, y = get_batch(train_data, micro_batch, block_size, device)
                with autocast_ctx:
                    _, loss = model(x, y)
                    loss = loss / grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                train_loss_accum += loss.item()

            # Clip + step.
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Divergence check.
            if not math.isfinite(train_loss_accum):
                print(f"\nFAIL: loss diverged at step {step}: {train_loss_accum}")
                metrics.set_status("diverged")
                metrics.finalize()
                return 1

            tokens_seen += tokens_per_step
            last_step_time = time.monotonic() - t_step

            # Eval.
            val_loss: float | None = None
            if (step + 1) % eval_every == 0 or step == max_steps - 1:
                val_loss = evaluate(
                    model, val_data, micro_batch, block_size, eval_tokens,
                    device, autocast_ctx,
                )
                last_val_loss = val_loss

            metrics.log_step(tokens_seen, train_loss_accum, val_loss)

            # Console line.
            extras = f"  val={val_loss:.4f}" if val_loss is not None else ""
            print(f"  step={step + 1:>3d}/{max_steps}  lr={lr:.2e}  "
                  f"train={train_loss_accum:.4f}{extras}  "
                  f"step_time={last_step_time:.2f}s")

            # Dry run early exit (we keep MAX_STEPS as set by --max-steps already).
            # No special handling needed here.

    except torch.cuda.OutOfMemoryError as e:
        print(f"\nFAIL: OOM at step {step}: {e}", file=sys.stderr)
        metrics.set_status("oom")
        if device.type == "cuda":
            metrics.set_peak_gpu_mem(torch.cuda.max_memory_allocated())
        metrics.finalize()
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        metrics.set_status("crashed")
        metrics.finalize()
        return 130

    wall = time.monotonic() - t_start
    if device.type == "cuda":
        metrics.set_peak_gpu_mem(torch.cuda.max_memory_allocated())

    # Dry-run projection.
    if is_dry_run:
        # Use the last-step time as the throughput proxy.
        if last_step_time and last_step_time > 0:
            tps = tokens_per_step / last_step_time
            print(f"\n--- DRY RUN PROFILE ---")
            print(f"  measured tokens/sec ({cfg.name}, last step): {tps:,.0f}")
            if device.type == "cuda":
                peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                print(f"  peak GPU memory: {peak_mb:.1f} MB")
            full_steps = 92  # spec default for 2 epochs
            print(f"  projected full-run wall times (FLOPs ~ N * tokens, {full_steps} steps):")
            projections = project_wall_times(cfg.name, tps, full_steps, tokens_per_step)
            for name, secs in projections.items():
                print(f"    {name:<10s}  ~{secs / 60:6.1f} min  ({secs / 3600:5.2f} hr)")

    # Final status + finalize.
    metrics.finalize(val_loss_final=last_val_loss)
    if not is_dry_run:
        metrics.set_status("ok")  # finalize already sets ok if status was 'running'

    # Checkpoint.
    if not is_dry_run and not args.no_checkpoint:
        ckpt_dir = Path(cfg.paths.checkpoints_dir) / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "model.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": max_steps,
            "config": asdict(cfg),
        }, ckpt_path)
        print(f"checkpoint: {ckpt_path}")

    print(f"\nDone in {wall:.1f}s. final val_loss={last_val_loss}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 5 training entry point.")
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--override", nargs="*", default=[],
                   help="key=value overrides, e.g. optimizer.lr=3e-3")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override schedule.max_steps for this run only")
    p.add_argument("--dry-run-profile", action="store_true",
                   help="run a short profiling pass and project wall times")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="skip saving the final checkpoint (used by lr_sweep)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    return train(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
