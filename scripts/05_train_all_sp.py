"""Train all 5 SP sizes with the winning LR from the LR sweep.

Reads configs/sp_winning_lr.txt, then runs sp_tiny -> sp_xl in sequence
with that LR overridden into each. Each run writes runs/{run_id}.json and
saves a final checkpoint. If one size OOMs or diverges, the rest still run
(the wrapper try/excepts and continues).

Usage:
    python scripts/05_train_all_sp.py
    python scripts/05_train_all_sp.py --sizes sp_large sp_xl   # subset re-run
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.train import main as train_main  # noqa: E402


SIZES_DEFAULT = ["sp_tiny", "sp_small", "sp_medium", "sp_large", "sp_xl"]


def run_one(size: str, lr: float, runs_dir: Path) -> dict:
    """Run a single size. Returns the parsed metrics dict (or a stub on failure)."""
    cfg_path = ROOT / "configs" / f"{size}.yaml"
    if not cfg_path.exists():
        return {"size": size, "status": "missing_config", "val_loss_final": None}

    existing = set(runs_dir.glob("*.json"))

    argv = [
        "--config", str(cfg_path),
        "--override", f"optimizer.lr={lr}",
    ]

    err: str | None = None
    try:
        rc = train_main(argv)
    except torch.cuda.OutOfMemoryError as e:
        err = f"OOM: {e}"
        rc = 1
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        rc = 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    new = sorted(set(runs_dir.glob("*.json")) - existing, key=lambda p: p.stat().st_mtime)
    if not new:
        return {
            "size": size, "status": "missing",
            "val_loss_final": None, "error": err,
        }
    data = json.loads(new[-1].read_text(encoding="utf-8"))
    return {
        "size": size,
        "status": data.get("status"),
        "val_loss_final": data.get("val_loss_final"),
        "n_params": data.get("n_params"),
        "wall_seconds": data.get("wall_seconds"),
        "tokens_per_second_avg": data.get("tokens_per_second_avg"),
        "peak_gpu_mem_bytes": data.get("peak_gpu_mem_bytes"),
        "run_id": data.get("run_id"),
        "rc": rc,
        "error": err,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 7: train all SP sizes.")
    p.add_argument("--winner-file", default="configs/sp_winning_lr.txt")
    p.add_argument("--sizes", nargs="+", default=SIZES_DEFAULT)
    p.add_argument("--runs-dir", default="runs")
    args = p.parse_args(argv)

    winner_file = Path(args.winner_file)
    if not winner_file.exists():
        print(f"ERROR: {winner_file} not found. Run Phase 6 (lr_sweep) first.",
              file=sys.stderr)
        return 1
    lr = float(winner_file.read_text(encoding="utf-8").strip())
    if lr <= 0:
        print(f"ERROR: invalid LR {lr} in {winner_file}", file=sys.stderr)
        return 1

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Phase 7: SP scaling runs at LR = {lr:.0e}")
    print(f"  sizes = {args.sizes}")
    print(f"  runs_dir = {runs_dir}")
    print("=" * 70)

    results: list[dict] = []
    t_start = time.monotonic()
    for i, size in enumerate(args.sizes):
        print(f"\n{'#' * 70}")
        print(f"# [{i + 1}/{len(args.sizes)}] {size}")
        print(f"{'#' * 70}\n")
        r = run_one(size, lr, runs_dir)
        results.append(r)
        v = r.get("val_loss_final")
        print(f"\n  -> {size}: status={r['status']}  "
              f"val_loss_final={v if v is None else f'{v:.4f}'}  "
              f"wall={r.get('wall_seconds', 0) or 0:.0f}s")

    wall = time.monotonic() - t_start

    # Summary table.
    print()
    print("=" * 88)
    print("PHASE 7 SCALING SUMMARY")
    print("=" * 88)
    print(f"{'size':<10}  {'n_params':>10}  {'val_loss':>10}  {'wall':>8}  "
          f"{'tok/s':>10}  {'peak_MB':>8}  {'status':>10}")
    print("-" * 88)
    for r in results:
        n = r.get("n_params") or 0
        v = r.get("val_loss_final")
        v_str = f"{v:.4f}" if isinstance(v, (int, float)) else ","
        w = r.get("wall_seconds") or 0
        tps = r.get("tokens_per_second_avg") or 0
        mem = (r.get("peak_gpu_mem_bytes") or 0) / (1024 * 1024)
        print(f"{r['size']:<10}  {n:>10,}  {v_str:>10}  {w:>7.0f}s  "
              f"{tps:>10,.0f}  {mem:>7.0f}M  {r['status']:>10}")
    print()
    print(f"Total wall time: {wall / 60:.1f} min")

    # Write aggregated summary.
    summary_path = runs_dir / "sp_scaling_summary.json"
    summary_path.write_text(json.dumps({
        "winning_lr": lr,
        "total_wall_seconds": wall,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")

    # Exit non-zero if any failed.
    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        print(f"\nWARNING: {len(failed)} run(s) did not complete cleanly: "
              f"{[r['size'] for r in failed]}")
        return 2  # distinct exit code: completed but with failures
    print("\nAll 5 SP scaling runs completed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
