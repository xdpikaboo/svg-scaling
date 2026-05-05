"""LR sweep on sp_tiny over 7 log-spaced learning rates.

For each LR, runs sp_tiny to completion, captures the final val loss, and
writes:

    runs/sweep_sp_tiny/{run_id}.json        one per LR
    runs/sweep_sp_tiny/sweep_results.json   aggregated table
    configs/sp_winning_lr.txt               single float, consumed by the
                                            scaling-run wrapper

Diverged or OOM runs are recorded with val_loss_final=inf and skipped when
picking the winner. The high-LR end of the sweep (3e-2, 1e-1) is *expected*
to diverge; that's useful signal, not a failure.

CLI:
    python -m src.lr_sweep --config configs/sp_tiny.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from src.train import main as train_main


# 7 log-spaced LRs LRS_DEFAULT = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def run_one_lr(config_path: str, lr: float, output_dir: Path) -> dict[str, Any]:
    """Invoke train.main once for a given LR. Returns a result dict."""
    existing = set(output_dir.glob("*.json"))

    argv = [
        "--config", config_path,
        "--no-checkpoint",
        "--override",
        f"optimizer.lr={lr}",
        f"paths.runs_dir={output_dir.as_posix()}",
    ]

    rc: int = 0
    err_msg: str | None = None
    try:
        rc = train_main(argv)
    except torch.cuda.OutOfMemoryError as e:
        err_msg = f"OOM: {e}"
        rc = 1
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        rc = 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Identify the run JSON written by this invocation (newly created).
    new_files = sorted(
        set(output_dir.glob("*.json")) - existing,
        key=lambda p: p.stat().st_mtime,
    )
    if not new_files:
        return {
            "lr": lr, "rc": rc, "status": "missing",
            "val_loss_final": math.inf, "run_id": None,
            "error": err_msg or "no metrics json produced",
            "json_path": None,
        }

    json_path = new_files[-1]
    data = json.loads(json_path.read_text(encoding="utf-8"))
    val_final = data.get("val_loss_final")
    if val_final is None or not isinstance(val_final, (int, float)) or not math.isfinite(val_final):
        val_final = math.inf
    return {
        "lr": lr, "rc": rc,
        "status": data.get("status", "unknown"),
        "val_loss_final": float(val_final),
        "run_id": data.get("run_id"),
        "error": err_msg,
        "json_path": str(json_path),
    }


def pick_winner(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the LR with min finite val_loss_final among ok runs.
    Tie-break (within 0.01 of best): prefer the *smaller* LR for stability."""
    finite = [
        r for r in results
        if r["status"] == "ok" and math.isfinite(r["val_loss_final"])
    ]
    if not finite:
        return None
    best = min(finite, key=lambda r: r["val_loss_final"])
    near = [r for r in finite if r["val_loss_final"] - best["val_loss_final"] < 0.01]
    return min(near, key=lambda r: r["lr"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 6 LR sweep on sp_tiny.")
    p.add_argument("--config", default="configs/sp_tiny.yaml")
    p.add_argument("--output-dir", default="runs/sweep_sp_tiny")
    p.add_argument("--lrs", nargs="+", type=float, default=LRS_DEFAULT)
    p.add_argument("--winner-file", default="configs/sp_winning_lr.txt")
    args = p.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"LR sweep: config={args.config}")
    print(f"          {len(args.lrs)} LRs: {args.lrs}")
    print(f"          output_dir={output_dir}")
    print()

    results: list[dict[str, Any]] = []
    for i, lr in enumerate(args.lrs):
        print("=" * 60)
        print(f"[{i + 1}/{len(args.lrs)}] LR = {lr:.0e}")
        print("=" * 60)
        r = run_one_lr(args.config, lr, output_dir)
        results.append(r)
        v = f"{r['val_loss_final']:.4f}" if math.isfinite(r["val_loss_final"]) else "inf"
        print(f"\n  -> val_loss_final={v}  status={r['status']}\n")

    # Persist aggregated table.
    results_path = output_dir / "sweep_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Pretty summary.
    print()
    print("=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(f"{'LR':>10}  {'val_loss':>10}  {'status':>10}  {'run_id':>30}")
    print("-" * 70)
    for r in results:
        v = f"{r['val_loss_final']:.4f}" if math.isfinite(r['val_loss_final']) else "inf"
        rid = r.get("run_id") or ","
        print(f"{r['lr']:>10.0e}  {v:>10}  {r['status']:>10}  {rid:>30}")
    print()

    # Winner.
    winner = pick_winner(results)
    if winner is None:
        print("FAILED: no run completed with finite val_loss_final.")
        Path(args.winner_file).write_text("0.0\n", encoding="utf-8")
        return 1

    win_lr = winner["lr"]
    print(f"Winning LR: {win_lr:.0e}  "
          f"(val_loss_final={winner['val_loss_final']:.4f}, status={winner['status']})")

    # Sanity warnings.
    if win_lr <= 1e-4 or win_lr >= 1e-1:
        print(f"WARNING: winning LR {win_lr:.0e} is at a sweep extreme, "
              "investigate before kicking off Phase 7.", file=sys.stderr)
    elif abs(math.log10(win_lr) - math.log10(3e-3)) < 0.5:
        print("(Matches spec's expected ~3e-3 region.)")

    Path(args.winner_file).write_text(f"{win_lr}\n", encoding="utf-8")
    print(f"Written: {args.winner_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
