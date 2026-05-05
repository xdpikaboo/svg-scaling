"""Train the best model (configs/best.yaml = mup_xl x 3 epochs).

Wrapper around src/train.py that records the resulting run_id to
runs/best_run_id.txt so the generation and eval steps can locate the
checkpoint deterministically.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import main as train_main  # noqa: E402


def main() -> int:
    cfg_path = ROOT / "configs" / "best.yaml"
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found", file=sys.stderr)
        return 1
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    existing = set(runs_dir.glob("*.json"))

    print("=" * 70)
    print("Best model: muP-XL x 3 epochs (~138 steps, ~72M tokens seen)")
    print("=" * 70)

    t0 = time.monotonic()
    rc = train_main(["--config", str(cfg_path)])
    wall = time.monotonic() - t0

    if rc != 0:
        print(f"\ntrain.main exited rc={rc} after {wall:.0f}s", file=sys.stderr)
        return rc

    new = sorted(set(runs_dir.glob("*.json")) - existing, key=lambda p: p.stat().st_mtime)
    if not new:
        print("WARNING: no new run JSON produced", file=sys.stderr)
        return 1

    metrics = json.loads(new[-1].read_text(encoding="utf-8"))
    run_id = metrics.get("run_id")
    print(f"\nbest run_id: {run_id}")
    print(f"  val_loss_final = {metrics.get('val_loss_final')}")
    print(f"  wall = {metrics.get('wall_seconds')}s")
    print(f"  status = {metrics.get('status')}")

    (ROOT / "runs" / "best_run_id.txt").write_text(f"{run_id}\n", encoding="utf-8")
    print(f"\nrun id saved: runs/best_run_id.txt")
    print(f"Next: python scripts/09_generate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
